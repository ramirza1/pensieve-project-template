from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from registry import load_registry, save_registry, file_fingerprint, is_unchanged
from extract_docx import extract_docx_blocks
from chunking import chunk_blocks

def safe_id(s: str, max_len: int = 120) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return s[:max_len] if s else "no_h3"

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def parse_paper_title_authors_year(h3: str) -> tuple[str, str, str]:
    """
    Very lightweight parser for strings like:
    'In-Your-Face Politics: ... (Mutz, 2015)'
    Returns: (paper_title, authors, year)
    """
    s = (h3 or "").strip()
    if not s:
        return ("", "", "")

    year = ""
    m = YEAR_RE.search(s)
    if m:
        year = m.group(0)

    authors = ""
    if "(" in s and ")" in s:
        last_paren = s.rfind("(")
        if last_paren != -1 and s.endswith(")"):
            inside = s[last_paren + 1 : -1].strip()
            if YEAR_RE.search(inside):
                authors = inside.replace(year, "").strip(" ,;")
                title = s[:last_paren].strip()
                return (title, authors, year)

    return (s, authors, year)


def load_config() -> Dict[str, Any]:
    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def docx_to_block_dicts(docx_path: Path) -> List[Dict[str, Any]]:
    blocks = extract_docx_blocks(docx_path)
    return [
        {
            "heading_path": b.heading_path,
            "heading_levels": getattr(b, "heading_levels", []),
            "text": b.text,
            "para_index": b.para_index,
            "style": b.style,
        }
        for b in blocks
    ]


def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    load_dotenv()
    cfg = load_config()

    target = (os.getenv("PENSIEVE_TARGET", "") or "").strip().lower()

    repo = Path(__file__).resolve().parent.parent

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (repo / pp).resolve()

    notes_dir = _abs(cfg["paths"]["notes_dir"])

    # ---- Chroma dir (IDENTICAL LOGIC to index_papers.py) ----
    chosen_chroma = (
        (os.getenv("PENSIEVE_CHROMA_DIR") or "").strip()
        or (cfg.get("paths", {}) or {}).get("chroma_dir")  # optional future override
        or (
            (cfg.get("paths", {}) or {}).get("chroma_dir_server")
            if target == "server"
            else (cfg.get("paths", {}) or {}).get("chroma_dir_local")
        )
    )
    if not chosen_chroma:
        raise KeyError(
            "No chroma dir configured. Set PENSIEVE_CHROMA_DIR or add "
            "paths.chroma_dir(_local/_server) in config.yaml"
        )
    chroma_dir = _abs(chosen_chroma)

    # ---- Processed dir (safe to sync) ----
    chosen_processed = (os.getenv("PENSIEVE_PROCESSED_DIR") or "").strip() or cfg["paths"]["processed_dir"]
    processed_dir = _abs(chosen_processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---- Registry path ----
    chosen_registry = (
        (os.getenv("PENSIEVE_REGISTRY_PATH") or "").strip()
        or (
            cfg["paths"]["registry_server"]
            if target == "server"
            else cfg["paths"]["registry_local"]
        )
    )
    registry_path = _abs(chosen_registry)
    registry = load_registry(registry_path)

    embed_model = cfg.get("embedding_model", "text-embedding-3-small")

    print("Using target:", target or "(default)")
    print("notes_dir:", notes_dir)
    print("processed_dir:", processed_dir)
    print("chroma_dir:", chroma_dir)
    print("registry_path:", registry_path)

    indexing_cfg = cfg.get("indexing", {})

    notes_cfg = indexing_cfg.get("notes", indexing_cfg)  # supports old flat config too
    chunk_size = int(notes_cfg.get("chunk_size", 1400))
    chunk_overlap = int(notes_cfg.get("chunk_overlap", 150))

    force_reindex = bool(indexing_cfg.get("force_reindex_notes", indexing_cfg.get("force_reindex", False)))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    oa = OpenAI(api_key=api_key)

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    col = chroma.get_or_create_collection(name="notes")
    col_docs = chroma.get_or_create_collection(name="notes_docs")

    docx_files = list(notes_dir.rglob("*.docx"))
    if not docx_files:
        print(f"No .docx files found under {notes_dir.resolve()}")
        return

    for docx_path in docx_files:
        rel = docx_path.relative_to(notes_dir)
        rel_norm = str(rel).replace("\\", "/")
        print(f"\nIndexing: {rel_norm}")

        fp = file_fingerprint(docx_path)
        reg_key = f"notes::{rel_norm}"
        if (not force_reindex) and is_unchanged(registry, reg_key, fp):
            print("  Skipping (unchanged since last index).")
            continue

        blocks = docx_to_block_dicts(docx_path)

        # Save extracted blocks for debugging
        out_json = processed_dir / (docx_path.stem + ".blocks.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)

        chunks = chunk_blocks(blocks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Build parallel lists:
        # - store_texts: what you store/display
        # - embed_texts_list: what you embed (context prefix + chunk)
        store_texts: List[str] = []
        embed_texts_list: List[str] = []
        chunk_metas: List[Dict[str, Any]] = []

        for c in chunks:
            chunk_text = (c.get("text") or "").strip()
            if not chunk_text:
                continue

            hp = c.get("heading_path") or []
            hl = c.get("heading_levels") or []

            def last_heading(level: int) -> str:
                for text, lvl in zip(reversed(hp), reversed(hl)):
                    if lvl == level:
                        return text
                return ""

            theme = last_heading(2)
            paper_heading = last_heading(3)
            subpaper = last_heading(4)
            paper_source = paper_heading or subpaper

            paper_title, paper_authors, paper_year = parse_paper_title_authors_year(paper_source)

            # Context prefix for EMBEDDING ONLY
            ctx_bits = []
            if theme:
                ctx_bits.append(f"Theme: {theme}")

            if paper_title or paper_authors or paper_year:
                p = paper_title if paper_title else "Paper"
                if paper_authors:
                    p += f" — {paper_authors}"
                if paper_year:
                    p += f" ({paper_year})"
                ctx_bits.append(p)

            # H3 defines the "doc" (one result per paper/section)
            doc_anchor = paper_heading or "(no h3)"
            doc_id = f"{rel_norm}::h3::{safe_id(doc_anchor)}"

            if hp:
                ctx_bits.append(" > ".join(hp))

            embed_text = ("\n".join(ctx_bits) + "\n\n" + chunk_text) if ctx_bits else chunk_text

            store_texts.append(chunk_text)
            embed_texts_list.append(embed_text)

            chunk_metas.append(
                {
                    "source_file": rel_norm,
                    "source_type": "note",
                    "heading_path": " > ".join(hp) if hp else "",
                    "theme": theme,
                    "paper_title": paper_title,
                    "paper_authors": paper_authors,
                    "paper_year": paper_year,
                    "start_para_index": c.get("start_para_index"),
                    "end_para_index": c.get("end_para_index"),
                    "doc_id": doc_id,
                    "doc_anchor": doc_anchor,
                }
            )

        if not store_texts:
            print("  No chunk texts produced; skipping file.")
            registry[reg_key] = {**fp, "chunks": 0, "note": "no_chunks"}
            save_registry(registry_path, registry)
            continue

        # Embed in batches
        batch = 96
        all_ids: List[str] = []
        all_embeds: List[List[float]] = []
        all_docs: List[str] = []
        all_metas: List[Dict[str, Any]] = []

        for i in range(0, len(embed_texts_list), batch):
            batch_texts = embed_texts_list[i : i + batch]
            embeds = embed_texts(oa, batch_texts, model=embed_model)

            for j, e in enumerate(embeds):
                chunk_ix = i + j
                cid = f"{rel_norm}::chunk::{chunk_ix}"

                all_ids.append(cid)
                all_embeds.append(e)
                all_docs.append(store_texts[chunk_ix])

                meta = dict(chunk_metas[chunk_ix])
                meta["chunk_index"] = chunk_ix
                all_metas.append(meta)

        # Remove any old chunks for this file so we don't keep stale chunks around
        col.delete(where={"source_file": rel_norm})

        # Upsert to Chroma
        col.upsert(
            ids=all_ids,
            embeddings=all_embeds,
            documents=all_docs,
            metadatas=all_metas,
        )

        # -------------------------
        # Doc-level index (one per H3)
        # -------------------------
        from collections import defaultdict

        doc_texts = defaultdict(list)
        doc_metas = {}

        for doc, meta in zip(all_docs, all_metas):
            did = meta.get("doc_id")
            if not did:
                continue
            doc_texts[did].append(doc)

            # Keep one "best" meta per doc (first seen is fine)
            if did not in doc_metas:
                doc_metas[did] = {
                    "source_file": meta.get("source_file"),
                    "source_type": "note",
                    "doc_id": did,
                    "doc_anchor": meta.get("doc_anchor", ""),
                    "theme": meta.get("theme", ""),
                    "paper_title": meta.get("paper_title", ""),
                    "paper_authors": meta.get("paper_authors", ""),
                    "paper_year": meta.get("paper_year", ""),
                    "heading_path": meta.get("heading_path", ""),
                }

        doc_ids = []
        doc_docs = []
        doc_embed_inputs = []
        doc_metadatas = []

        MAX_DOC_CHARS = 20000  # safe cap; embeddings don't need the entire section verbatim

        for did, parts in doc_texts.items():
            full = "\n\n".join(p.strip() for p in parts if p and p.strip())
            if not full:
                continue

            meta = doc_metas[did]
            # context prefix for embedding
            ctx_bits = []
            if meta.get("theme"):
                ctx_bits.append(f"Theme: {meta['theme']}")
            p = meta.get("paper_title") or meta.get("doc_anchor") or "Paper"
            if meta.get("paper_authors"):
                p += f" — {meta['paper_authors']}"
            if meta.get("paper_year"):
                p += f" ({meta['paper_year']})"
            ctx_bits.append(p)
            if meta.get("heading_path"):
                ctx_bits.append(meta["heading_path"])

            prefix = "\n".join(ctx_bits)
            embed_input = (prefix + "\n\n" + full) if prefix else full

            # cap size for embedding + UI storage
            full_capped = full[:MAX_DOC_CHARS]
            embed_capped = embed_input[:MAX_DOC_CHARS]

            doc_ids.append(did)
            doc_docs.append(full_capped)
            doc_embed_inputs.append(embed_capped)
            doc_metadatas.append(meta)

        # replace old doc-level entries for this file
        col_docs.delete(where={"source_file": rel_norm})

        # embed + upsert doc-level
        doc_embs = embed_texts(oa, doc_embed_inputs, model=embed_model)
        col_docs.upsert(ids=doc_ids, embeddings=doc_embs, documents=doc_docs, metadatas=doc_metadatas)
        print(f"  Added {len(doc_docs)} doc-level NOTE entries to collection 'notes_docs'.")

        registry[reg_key] = {**fp, "chunks": len(all_docs)}
        save_registry(registry_path, registry)

        print(f"  Added {len(all_docs)} chunks.")

    print("\nDone. Notes indexed into Chroma collection: 'notes'")


if __name__ == "__main__":
    main()