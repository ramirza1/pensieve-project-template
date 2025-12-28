from __future__ import annotations

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# -------------------------
# Config + utilities
# -------------------------

def load_config() -> Dict[str, Any]:
    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_id(s: str, max_len: int = 180) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]", "", s)
    return s[:max_len] if s else "no_id"

def short_hash(s: str, n: int = 16) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()[:n]

def unit_fingerprint(text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fingerprint based on (doc text + key meta). Stable across reruns.
    """
    base = {
        "source_file": meta.get("source_file", ""),
        "doc_id": meta.get("doc_id", ""),
        "doc_anchor": meta.get("doc_anchor", ""),
        "paper_title": meta.get("paper_title", ""),
        "paper_authors": meta.get("paper_authors", ""),
        "paper_year": meta.get("paper_year", ""),
        "theme": meta.get("theme", ""),
    }
    payload = (json.dumps(base, sort_keys=True) + "\n\n" + (text or "")).encode("utf-8", errors="ignore")
    return {
        "sha": hashlib.sha256(payload).hexdigest(),
        "meta": base,
        "text_len": len(text or ""),
    }

def read_cached_summary(cache_path: Path, fp: Dict[str, Any]) -> Optional[dict]:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (data or {}).get("fingerprint") != fp:
        return None
    return (data or {}).get("summary")

def write_cached_summary(cache_path: Path, fp: Dict[str, Any], summary: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fp,
        "saved_at": int(time.time()),
        "summary": summary,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def truncate_chars(s: str, max_chars: int) -> str:
    if not s:
        return s
    return s[:max_chars]

# -------------------------
# JSON parsing + formatting helpers
# -------------------------

def _parse_json_best_effort(s: str) -> Optional[dict]:
    """
    Attempts to parse JSON even when the model wraps it in ```json fences,
    adds leading/trailing text, uses smart quotes, or leaves trailing commas.
    """
    if not s:
        return None

    raw = s.strip()

    # Strip fenced code blocks if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    # If there's surrounding text, extract first {...} block
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        raw = raw[i : j + 1].strip()

    # Replace smart quotes
    raw = raw.replace("“", '"').replace("”", '"').replace("’", "'")

    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def abstract_looks_like_json(abstract: str) -> bool:
    a = (abstract or "").strip()
    if not a:
        return False
    if a.startswith("{") or a.startswith("```"):
        return True
    head = a[:400]
    if '"summary_abstract"' in head or '"key_points"' in head or '"concepts"' in head:
        return True
    return False


def normalize_core(core: dict) -> dict:
    """
    Ensure expected keys exist and list fields are lists.
    """
    core = dict(core or {})

    if "summary_abstract" not in core or core["summary_abstract"] is None:
        core["summary_abstract"] = ""

    for k in ["key_points", "concepts", "open_questions", "next_steps"]:
        v = core.get(k)
        if v is None:
            core[k] = []
        elif not isinstance(v, list):
            core[k] = [str(v)]

    return core

def summary_abstract_only(summary_obj: dict) -> str:
    return (summary_obj.get("summary_abstract") or "").strip()

def format_note_summary_md(summary_obj: dict) -> str:
    """
    Convert structured note summary into markdown suitable for Streamlit rendering.
    """
    summary_obj = summary_obj or {}
    abstract = (summary_obj.get("summary_abstract") or "").strip()

    def bullets(items) -> str:
        if not isinstance(items, list):
            return ""
        items2 = [str(x).strip() for x in items if str(x).strip()]
        return "\n".join(f"- {x}" for x in items2)

    parts: List[str] = []
    if abstract:
        parts.append(abstract)

    for key, label in [
        ("key_points", "Key points"),
        ("concepts", "Concepts"),
        ("open_questions", "Open questions"),
        ("next_steps", "Next steps"),
    ]:
        b = bullets(summary_obj.get(key))
        if b:
            parts.append(f"**{label}**\n{b}")

    return "\n\n".join(parts).strip()


# -------------------------
# Summarization prompt
# -------------------------

PROMPT_SCHEMA = """Return ONLY valid JSON with the following keys:
- summary_abstract: string (90–150 words, neutral academic tone; summarize the note section as if it were an abstract)
- key_points: array of 4–10 short strings
- concepts: array of 5–15 keywords/phrases (1–4 words each)
- open_questions: array of 0–6 short strings
- next_steps: array of 0–6 short strings

Rules:
- Do NOT invent details not supported by the text.
- Prefer faithful paraphrase; preserve the author’s intent.
- No citations, no quotes.
- No phrase longer than 12 consecutive words should match the source text.
- Do not paraphrase long passages; synthesize across sections
- Avoid distinctive terminology unless it is a named concept (e.g., ‘argumentative theory of reasoning’).
"""

def summarize_note_section_json(
    client: OpenAI,
    *,
    header: str,
    text: str,
    model: str = "gpt-4.1-mini",
    max_input_chars: int = 120_000,
) -> dict:
    content = truncate_chars(text, max_input_chars)

    messages = [
        {
            "role": "system",
            "content": "You summarize research notes into concise, structured, abstract-style summaries.",
        },
        {
            "role": "user",
            "content": f"{PROMPT_SCHEMA}\n\nSECTION HEADER:\n{header}\n\nNOTES TEXT:\n{content}",
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    out = (resp.choices[0].message.content or "").strip()

    parsed = _parse_json_best_effort(out)
    if parsed:
        return normalize_core(parsed)

    return normalize_core({
        "summary_abstract": out[:2000].strip(),
        "key_points": [],
        "concepts": [],
        "open_questions": [],
        "next_steps": [],
        "_parse_error": True,
    })


# -------------------------
# Main pipeline
# -------------------------

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    cfg = load_config()

    target = (os.getenv("PENSIEVE_TARGET", "") or "").strip().lower()
    repo = Path(__file__).resolve().parent.parent

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (repo / pp)

    # ---- Chroma dir (IDENTICAL LOGIC to index_notes/index_papers) ----
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

    print("Using target:", target or "(default)")
    print("processed_dir:", processed_dir)
    print("chroma_dir:", chroma_dir)

    # ---- Summarization config ----
    summ_cfg = cfg.get("summarization", {})

    # Force regenerate everything regardless of cache
    force_resummarize = bool(summ_cfg.get("force_resummarize_notes", False))

    # Repair-only mode: only re-summarize cached units that are "bad"
    repair_bad_cached_only = bool(summ_cfg.get("repair_bad_cached_notes_only", True))

    gen_model = str(summ_cfg.get("model", os.getenv("SUMMARY_MODEL", "gpt-4.1-mini")))
    write_to_chroma = bool(summ_cfg.get("write_summaries_to_chroma", True))

    # Where we store summaries (JSON cache)
    summaries_dir = processed_dir / "note_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    embed_model = cfg.get("embedding_model", "text-embedding-3-small")

    oa = OpenAI(api_key=api_key)

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    notes_docs = chroma.get_or_create_collection(name="notes_docs")
    col_sum = chroma.get_or_create_collection(name="note_summaries") if write_to_chroma else None

    def embed_texts(texts: List[str]) -> List[List[float]]:
        resp = oa.embeddings.create(model=embed_model, input=texts)
        return [d.embedding for d in resp.data]

    # Pull all doc-level note units
    batch = 200
    total = 0
    done = 0
    reused = 0
    repaired = 0
    skipped_short = 0
    errors = 0
    t0 = time.time()

    def should_regenerate_from_cached(cached: dict) -> bool:
        if not cached:
            return True
        if cached.get("_parse_error"):
            return True
        if abstract_looks_like_json(cached.get("summary_abstract", "")):
            return True
        if "summary_abstract" not in cached:
            return True
        return False

    # 1) Pull all ids
    all_ids_payload = notes_docs.get(include=[])
    all_ids = all_ids_payload.get("ids") or []

    if not all_ids:
        print("No doc-level notes found in 'notes_docs'. Run index_notes.py first.")
        return

    # 2) Process in batches by ids
    for i in range(0, len(all_ids), batch):
        batch_ids = all_ids[i:i+batch]
        got = notes_docs.get(
            ids=batch_ids,
            include=["documents", "metadatas"],
        )

        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []

        for did, doc_text, meta in zip(ids, docs, metas):
            total += 1
            doc_text = doc_text or ""
            meta = meta or {}

            source_file = meta.get("source_file", "")
            doc_anchor = meta.get("doc_anchor", "")
            theme = meta.get("theme", "")
            paper_title = meta.get("paper_title", "")
            paper_authors = meta.get("paper_authors", "")
            paper_year = meta.get("paper_year", "")

            header_bits = []
            if source_file:
                header_bits.append(f"Source: {source_file}")
            if theme:
                header_bits.append(f"Theme: {theme}")

            title_line = paper_title or doc_anchor or "(notes section)"
            if paper_authors:
                title_line += f" — {paper_authors}"
            if paper_year:
                title_line += f" ({paper_year})"
            header_bits.append(title_line)

            header = "\n".join(header_bits)

            fp = unit_fingerprint(doc_text, meta)
            slug = safe_id(meta.get("paper_title") or meta.get("doc_anchor") or "note", max_len=60)
            cache_path = summaries_dir / f"{slug}__{short_hash(did)}.json"

            cached = read_cached_summary(cache_path, fp)

            # Decide whether to regenerate
            regen = False
            if force_resummarize:
                regen = True
            else:
                if cached is None:
                    regen = True
                elif repair_bad_cached_only and should_regenerate_from_cached(cached):
                    regen = True
                else:
                    regen = False

            # Short notes: write empty-ish summary + continue
            if len(doc_text.strip()) < 200:
                summary_obj = normalize_core({
                    "unit_type": "note_section",
                    "doc_id": did,
                    "source_file": source_file,
                    "doc_anchor": doc_anchor,
                    "theme": theme,
                    "paper_title": paper_title,
                    "paper_authors": paper_authors,
                    "paper_year": paper_year,
                    "summary_abstract": "",
                    "key_points": [],
                    "concepts": [],
                    "open_questions": [],
                    "next_steps": [],
                    "note": "too_short",
                })
                skipped_short += 1
                write_cached_summary(cache_path, fp, summary_obj)

                # still write abstract to chroma (empty is fine)
                if write_to_chroma and col_sum is not None:
                    abstract = (summary_obj.get("summary_abstract") or "").strip()
                    doc_for_chroma = abstract
                    embed_input = (header + "\n\n" + abstract).strip()

                    if embed_input:
                        emb = embed_texts([embed_input])[0]
                        sid = f"{did}::summary"
                        col_sum.delete(ids=[sid])
                        col_sum.upsert(
                            ids=[sid],
                            embeddings=[emb],
                            documents=[doc_for_chroma],
                            metadatas=[{
                                "unit_type": "note_section",
                                "doc_id": did,
                                "source_file": source_file,
                                "doc_anchor": doc_anchor,
                                "theme": theme,
                                "paper_title": paper_title,
                                "paper_authors": paper_authors,
                                "paper_year": paper_year,
                            }],
                        )
                continue

            label = (paper_title or doc_anchor or did)[:80]

            # Reuse cached path (but ensure schema + write markdown to Chroma)
            if not regen and cached is not None:
                summary_obj = normalize_core(dict(cached))

                # If abstract actually contains json, force regen (repair mode)
                if abstract_looks_like_json(summary_obj.get("summary_abstract", "")) and repair_bad_cached_only:
                    regen = True
                else:
                    # if normalization changed anything, rewrite cache
                    if summary_obj != cached:
                        write_cached_summary(cache_path, fp, summary_obj)
                        repaired += 1

                    if write_to_chroma and col_sum is not None:
                        abstract = (summary_obj.get("summary_abstract") or "").strip()
                        doc_for_chroma = abstract
                        embed_input = (header + "\n\n" + abstract).strip()
                        if embed_input:
                            emb = embed_texts([embed_input])[0]
                            sid = f"{did}::summary"
                            col_sum.delete(ids=[sid])
                            col_sum.upsert(
                                ids=[sid],
                                embeddings=[emb],
                                documents=[doc_for_chroma],  # markdown for clean UI rendering
                                metadatas=[{
                                    "unit_type": "note_section",
                                    "doc_id": did,
                                    "source_file": source_file,
                                    "doc_anchor": doc_anchor,
                                    "theme": theme,
                                    "paper_title": paper_title,
                                    "paper_authors": paper_authors,
                                    "paper_year": paper_year,
                                }],
                            )
                    reused += 1
                    if reused % 25 == 0:
                        print(f"[{total}/{len(all_ids)}] reused cached: {reused}", flush=True)
                    continue

            # Regenerate path
            print(f"[{total}/{len(all_ids)}] {label} ...", flush=True)

            try:
                core = summarize_note_section_json(
                    oa,
                    header=header,
                    text=doc_text,
                    model=gen_model,
                )
            except Exception as e:
                errors += 1
                print(f"  ! ERROR summarizing {label}: {e}", flush=True)
                continue

            summary_obj = normalize_core({
                "unit_type": "note_section",
                "doc_id": did,
                "source_file": source_file,
                "doc_anchor": doc_anchor,
                "theme": theme,
                "paper_title": paper_title,
                "paper_authors": paper_authors,
                "paper_year": paper_year,
                **core,
            })

            write_cached_summary(cache_path, fp, summary_obj)
            done += 1
            print(f"  ✓ wrote {cache_path.name}", flush=True)

            if write_to_chroma and col_sum is not None:
                abstract = (summary_obj.get("summary_abstract") or "").strip()
                doc_for_chroma = abstract
                embed_input = (header + "\n\n" + abstract).strip()

                if embed_input:
                    emb = embed_texts([embed_input])[0]
                    sid = f"{did}::summary"
                    col_sum.delete(ids=[sid])
                    col_sum.upsert(
                        ids=[sid],
                        embeddings=[emb],
                        documents=[doc_for_chroma],
                        metadatas=[{
                            "unit_type": "note_section",
                            "doc_id": did,
                            "source_file": source_file,
                            "doc_anchor": doc_anchor,
                            "theme": theme,
                            "paper_title": paper_title,
                            "paper_authors": paper_authors,
                            "paper_year": paper_year,
                        }],
                    )

    elapsed = time.time() - t0
    print(
        f"\nDone.\n"
        f"- regenerated: {done}\n"
        f"- reused cached: {reused}\n"
        f"- repaired cached schema (no LLM): {repaired}\n"
        f"- skipped short: {skipped_short}\n"
        f"- errors: {errors}\n"
        f"Total processed {total}. Output: {summaries_dir} | elapsed={elapsed:.1f}s"
    )

if __name__ == "__main__":
    main()