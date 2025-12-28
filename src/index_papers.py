from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from chunking import chunk_blocks
from pdf_extract import extract_pdf_pages, cut_after_references, extract_title_authors_from_first_page
from registry import load_registry, save_registry, file_fingerprint, is_unchanged

import re

ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
STATE_RE = re.compile(r"\b[A-Z]{2}\b")  # crude but works for "PA"
AFFIL_RE = re.compile(
    r"\b(University|Department|School|Institute|Center|Centre|Laboratory|Lab|College)\b",
    re.IGNORECASE,
)

# Common “body text starts here” markers in extracted first pages
BODY_START_RE = re.compile(
    r"\b(ABSTRACT|Introduction|IN\s+THIS\s+|IN\s+THE\s+END\s+WE|We\s+study|We\s+argue|This\s+paper|Keywords|People\s+who)\b",
    re.IGNORECASE,
)

def clean_authors(authors_raw: str) -> str:
    """
    Heuristic cleanup for first-page extracted 'authors'.
    Goals:
    - remove affiliation/location tails (e.g., Philadelphia, PA 19104)
    - stop before body text accidentally appended
    - cap length and keep only plausible author tokens
    """
    if not authors_raw:
        return ""

    s = " ".join(authors_raw.split())  # collapse whitespace
    s = s.strip(" ,;:-")

    # 1) If body-text marker appears, cut everything after it
    m = BODY_START_RE.search(s)
    if m:
        s = s[: m.start()].strip(" ,;:-")

    # 2) If there's a ZIP code anywhere, cut from the comma/space before it
    m = ZIP_RE.search(s)
    if m:
        cut = m.start()
        # walk backwards to a natural separator
        sep = max(s.rfind(",", 0, cut), s.rfind("  ", 0, cut), s.rfind(" ", 0, cut))
        s = s[: sep].strip(" ,;:-") if sep != -1 else s[:cut].strip(" ,;:-")

    # 3) If we see a pattern like "City, ST" at the end, remove it
    # (Example: "Philadelphia, PA")
    # We'll remove the last ", XX" chunk if it's at the end or near-end
    parts = [p.strip() for p in s.split(",")]
    if len(parts) >= 2:
        last = parts[-1]
        # last part is just a state code (PA) or looks like state + something
        if STATE_RE.fullmatch(last) or STATE_RE.match(last):
            parts = parts[:-1]
            s = ", ".join(parts).strip(" ,;:-")

    # 4) Remove affiliation-y tails if they slipped in (keep left-most bit)
    # If affiliation word appears, cut from its start.
    m = AFFIL_RE.search(s)
    if m:
        s = s[: m.start()].strip(" ,;:-")

    # 5) Hard cap: if extraction still grabbed too much, trim to ~120 chars
    if len(s) > 120:
        s = s[:120].rsplit(" ", 1)[0].strip(" ,;:-")

    return s

def load_config() -> Dict[str, Any]:
    # Repo root: .../Pensieve (this file lives in /src typically)
    here = Path(__file__).resolve().parent.parent
    cfg_path = here / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def truncate_to_tokens(text: str, max_tokens: int = 6500) -> str:
    if not text:
        return text
    max_chars = max_tokens * 3
    return text[:max_chars]

def embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    load_dotenv()
    cfg = load_config()

    target = (os.getenv("PENSIEVE_TARGET", "") or "").strip().lower()

    # Resolve repo root so relative paths behave the same everywhere
    repo = Path(__file__).resolve().parent.parent

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (repo / pp)

    papers_dir = _abs(cfg["paths"]["papers_dir"])

    # Chroma dir selection logic (same as streamlit_app.py)
    chosen_chroma = (
        (os.getenv("PENSIEVE_CHROMA_DIR") or "").strip()
        or (cfg.get("paths", {}) or {}).get("chroma_dir")  # optional override if you add it later
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

    # Processed dir can remain in OneDrive (safe to sync)
    chosen_processed = (os.getenv("PENSIEVE_PROCESSED_DIR") or "").strip() or cfg["paths"]["processed_dir"]
    processed_dir = _abs(chosen_processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pdf_meta_cache_dir = processed_dir / "pdf_meta_cache"
    pdf_meta_cache_dir.mkdir(parents=True, exist_ok=True)

    chosen_registry = (
        (os.getenv("PENSIEVE_REGISTRY_PATH") or "").strip()
        or (
            cfg["paths"]["registry_server"]
            if target == "server"
            else cfg["paths"]["registry_local"]
        )
    )

    registry_path = _abs(chosen_registry)

    print("Using target:", target or "(default)")
    print("papers_dir:", papers_dir)
    print("processed_dir:", processed_dir)
    print("chroma_dir:", chroma_dir)
    print("registry_path:", registry_path)

    registry = load_registry(registry_path)

    indexing_cfg = cfg.get("indexing", {})
    papers_cfg = indexing_cfg.get("papers", indexing_cfg)
    chunk_size = int(papers_cfg.get("chunk_size", 1000))
    chunk_overlap = int(papers_cfg.get("chunk_overlap", 100))
    force_reindex = bool(indexing_cfg.get("force_reindex_papers", False))

    embed_model = cfg.get("embedding_model", "text-embedding-3-small")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    oa = OpenAI(api_key=api_key)

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    col = chroma.get_or_create_collection(name="papers")
    col_docs = chroma.get_or_create_collection(name="papers_docs")

    pdf_files = list(papers_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found under {papers_dir.resolve()}")
        return

    for pdf_path in pdf_files:
        rel = pdf_path.relative_to(papers_dir)
        rel_norm = str(rel).replace("\\", "/")
        print(f"\nIndexing PDF: {rel_norm}")

        fp = file_fingerprint(pdf_path)
        reg_key = f"papers::{rel_norm}"
        if (not force_reindex) and is_unchanged(registry, reg_key, fp):
            print("  Skipping (unchanged since last index).")
            continue

        # --- metadata: cache -> crossref -> pdf-meta -> first_page -> filename ---
        meta_best = extract_title_authors_from_first_page(
            pdf_path,
            cache_dir=pdf_meta_cache_dir,
            doi_lookup=True,
        )
        title = (meta_best.get("title") or "").strip()
        authors = (meta_best.get("authors") or "").strip()
        meta_source = (meta_best.get("meta_source") or "").strip()
        doi = (meta_best.get("doi") or "").strip()
        year = (meta_best.get("year") or "").strip()

        # Only clean heuristic / embedded sources (never touch Crossref)
        if meta_source in {"first_page", "pdf_meta"}:
            authors = clean_authors(authors)

        pages = extract_pdf_pages(pdf_path)
        pages = cut_after_references(pages)

        total_chars = sum(p["char_count"] for p in pages)
        text_pages = sum(1 for p in pages if p["has_text"])

        print(f"  Pages: {len(pages)} | pages-with-text: {text_pages} | total_chars: {total_chars}")

        if total_chars < 500:
            print("  Skipping (very little extractable text). Likely scanned PDF.")
            registry[reg_key] = {
                **fp,
                "chunks": 0,
                "meta_source": meta_source,
                "doi": doi,
                "year": year,
            }
            save_registry(registry_path, registry)
            continue

        # Treat each page as a "block" so chunking respects page boundaries
        page_chunks: List[Dict[str, Any]] = []
        for p in pages:
            if not p["text"]:
                continue

            page_blocks = [{
                "text": p["text"],
                "heading_path": [],
                "heading_levels": [],
                "para_index": p["page_num"],  # page number
            }]

            chunks = chunk_blocks(page_blocks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for c in chunks:
                c["page_num"] = p["page_num"]
                page_chunks.append(c)

        store_texts = [c["text"] for c in page_chunks]

        # Optional: embed with a context prefix (improves retrieval slightly)
        embed_texts_list = []
        for c in page_chunks:
            page_num = c.get("page_num")
            prefix_bits = []
            if title:
                prefix_bits.append(f"Title: {title}")
            if authors:
                prefix_bits.append(f"Authors: {authors}")
            if page_num:
                prefix_bits.append(f"Page: {page_num}")
            prefix = "\n".join(prefix_bits)
            embed_texts_list.append(prefix + "\n\n" + c["text"] if prefix_bits else c["text"])

        # embed in batches
        batch = 64
        ids: List[str] = []
        embs: List[List[float]] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        for i in range(0, len(embed_texts_list), batch):
            batch_texts = embed_texts_list[i:i+batch]
            batch_embs = embed_texts(oa, batch_texts, model=embed_model)

            for j, e in enumerate(batch_embs):
                chunk_ix = i + j
                cid = f"{rel_norm}::chunk::{chunk_ix}"
                ids.append(cid)
                embs.append(e)
                docs.append(store_texts[chunk_ix])

                hp = page_chunks[chunk_ix].get("heading_path") or []
                # store as a string (Streamlit expects string + .strip())
                hp_str = " > ".join([str(x).strip() for x in hp if str(x).strip()])

                metas.append(
                    {
                        "source_file": rel_norm,
                        "chunk_index": chunk_ix,
                        "source_type": "pdf",
                        "page_num": page_chunks[chunk_ix].get("page_num"),
                        "heading_path": hp_str,
                        "title": title,
                        "authors": authors,
                        "meta_source": meta_source,
                        "doi": doi,
                        "year": year,
                    }
                )

        # Replace old chunks for this file to avoid staleness
        col.delete(where={"source_file": rel_norm})

        col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        # -------------------------
        # Doc-level index (one per PDF)
        # -------------------------
        full = "\n\n".join([t.strip() for t in store_texts if t and t.strip()])
        if full:
            prefix_bits = []
            if title:
                prefix_bits.append(f"Title: {title}")
            if authors:
                prefix_bits.append(f"Authors: {authors}")
            prefix = "\n".join(prefix_bits)

            raw_full = full
            raw_embed = (prefix + "\n\n" + full) if prefix_bits else full

            doc_text = truncate_to_tokens(raw_full, max_tokens=6500)
            doc_embed_input = truncate_to_tokens(raw_embed, max_tokens=6500)

            doc_id = f"{rel_norm}::doc"
            doc_emb = embed_texts(oa, [doc_embed_input], model=embed_model)[0]

            col_docs.delete(where={"source_file": rel_norm})
            col_docs.upsert(
                ids=[doc_id],
                embeddings=[doc_emb],
                documents=[doc_text],
                metadatas=[{
                    "source_file": rel_norm,
                    "source_type": "pdf",
                    "doc_id": doc_id,
                    "heading_path": "",
                    "title": title,
                    "authors": authors,
                    "meta_source": meta_source,
                    "doi": doi,
                    "year": year,
                }],
            )
            print("  Added 1 doc-level PDF entry to collection 'papers_docs'.")
        print(f"  Added {len(docs)} chunks to collection 'papers'.")

        registry[reg_key] = {
            **fp, 
            "chunks": len(docs), 
            "meta_source": meta_source,
            "doi": doi, 
            "year": year
        }
        save_registry(registry_path, registry)

    print("\nDone. PDFs indexed into Chroma collection: 'papers'")


if __name__ == "__main__":
    main()