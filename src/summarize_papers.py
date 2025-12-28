from __future__ import annotations

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from pdf_extract import extract_pdf_pages, cut_after_references, extract_title_authors_from_first_page


# -------------------------
# Config + utilities
# -------------------------

def load_config() -> Dict[str, Any]:
    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_id(s: str, max_len: int = 160) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]", "", s)
    return s[:max_len] if s else "no_id"

def short_cache_name(rel_norm: str, max_prefix: int = 60) -> str:
    """
    Windows-safe cache filename: short readable prefix + stable hash.
    Prevents MAX_PATH issues and collisions.
    """
    prefix = safe_id(rel_norm, max_len=max_prefix)
    h = hashlib.sha256(rel_norm.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}__{h}.json"

def file_fingerprint_for_summary(pdf_path: Path) -> Dict[str, Any]:
    st = pdf_path.stat()
    payload = f"{pdf_path.name}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    return {
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "sha": hashlib.sha256(payload).hexdigest(),
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

def build_full_text_from_pdf(pdf_path: Path) -> str:
    pages = extract_pdf_pages(pdf_path)
    pages = cut_after_references(pages)
    full = "\n\n".join((p.get("text") or "").strip() for p in pages if (p.get("text") or "").strip())
    return full.strip()

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
    """
    Heuristic: detect when 'abstract' is actually a JSON blob that got stored as text.
    """
    a = (abstract or "").strip()
    if not a:
        return False
    if a.startswith("{") or a.startswith("```"):
        return True
    # common tell: JSON keys in the first ~400 chars
    head = a[:400]
    if '"summary_abstract"' in head or '"key_contributions"' in head or '"methods"' in head:
        return True
    return False


def normalize_core(core: dict) -> dict:
    """
    Ensure we use the UI-aligned schema:
      - findings (not main_findings)
    Also make sure list fields are lists.
    """
    core = dict(core or {})

    if "findings" not in core and "main_findings" in core:
        core["findings"] = core.get("main_findings") or []

    # normalize list fields
    for k in ["key_contributions", "methods", "findings", "limitations", "keywords"]:
        v = core.get(k)
        if v is None:
            core[k] = []
        elif not isinstance(v, list):
            core[k] = [str(v)]

    if "summary_abstract" not in core or core["summary_abstract"] is None:
        core["summary_abstract"] = ""

    return core

def summary_abstract_only(summary_obj: dict) -> str:
    return (summary_obj.get("summary_abstract") or "").strip()

def format_paper_summary_md(summary_obj: dict) -> str:
    """
    Convert structured summary into markdown suitable for Streamlit rendering.
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
        ("key_contributions", "Key contributions"),
        ("methods", "Methods"),
        ("findings", "Findings"),
        ("limitations", "Limitations"),
    ]:
        b = bullets(summary_obj.get(key))
        if b:
            parts.append(f"**{label}**\n{b}")

    return "\n\n".join(parts).strip()


# -------------------------
# Summarization (LLM)
# -------------------------

PROMPT_SCHEMA = """Return ONLY valid JSON with the following keys:
- summary_abstract: string (120–180 words, neutral academic tone)
- key_contributions: array of 3–6 short strings
- methods: array of 2–6 short strings (empty array if not clear)
- findings: array of 3–8 short strings (empty array if not clear)
- limitations: array of 1–5 short strings (empty array if not clear)
- keywords: array of 5–12 single- or two-word strings

Rules:
- Do NOT invent details not supported by the text.
- If uncertain, use cautious language (“suggests”, “appears”, “the paper argues”).
- Do NOT include citations or quotes.
- No phrase longer than 12 consecutive words should match the source text.
- Do not paraphrase long passages; synthesize across sections.
- Avoid distinctive terminology unless it is a named concept (e.g., ‘argumentative theory of reasoning’).
"""

def summarize_text_json(
    client: OpenAI,
    *,
    title: str,
    authors: str,
    year: str,
    doi: str,
    text: str,
    model: str = "gpt-4.1-mini",
    max_input_chars: int = 140_000,
) -> dict:
    meta_lines = []
    if title: meta_lines.append(f"Title: {title}")
    if authors: meta_lines.append(f"Authors: {authors}")
    if year: meta_lines.append(f"Year: {year}")
    if doi: meta_lines.append(f"DOI: {doi}")
    meta_header = "\n".join(meta_lines)

    content = truncate_chars(text, max_input_chars)

    messages = [
        {
            "role": "system",
            "content": "You write concise, neutral, academic abstract-style summaries from provided text and metadata.",
        },
        {
            "role": "user",
            "content": f"{PROMPT_SCHEMA}\n\nMETADATA:\n{meta_header}\n\nTEXT:\n{content}",
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

    # Fallback: treat output as plain abstract (not JSON)
    return normalize_core({
        "summary_abstract": out[:2000].strip(),
        "key_contributions": [],
        "methods": [],
        "findings": [],
        "limitations": [],
        "keywords": [],
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
    summ_cfg = cfg.get("summarization", {})

    target = (os.getenv("PENSIEVE_TARGET", "") or "").strip().lower()

    # Resolve repo root so relative paths behave the same everywhere
    repo = Path(__file__).resolve().parent.parent

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (repo / pp)

    # ---- Summarization toggles ----
    force_resummarize = bool(summ_cfg.get("force_resummarize_papers", False))
    repair_bad_cached_only = bool(summ_cfg.get("repair_bad_cached_papers_only", True))
    gen_model = str(summ_cfg.get("model", os.getenv("SUMMARY_MODEL", "gpt-4.1-mini")))
    write_to_chroma = bool(summ_cfg.get("write_summaries_to_chroma", True))

    # ---- Paths ----
    papers_dir = _abs(cfg["paths"]["papers_dir"])

    # Chroma dir (IDENTICAL LOGIC to index_papers/index_notes/summarize_notes)
    chosen_chroma = (
        (os.getenv("PENSIEVE_CHROMA_DIR") or "").strip()
        or (cfg.get("paths", {}) or {}).get("chroma_dir")  # optional override
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

    # Processed dir (safe to sync)
    chosen_processed = (os.getenv("PENSIEVE_PROCESSED_DIR") or "").strip() or cfg["paths"]["processed_dir"]
    processed_dir = _abs(chosen_processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Using target:", target or "(default)")
    print("papers_dir:", papers_dir)
    print("processed_dir:", processed_dir)
    print("chroma_dir:", chroma_dir)

    pdf_meta_cache_dir = processed_dir / "pdf_meta_cache"
    pdf_meta_cache_dir.mkdir(parents=True, exist_ok=True)

    # Where we store summaries (JSON cache)
    summaries_dir = processed_dir / "paper_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    embed_model = cfg.get("embedding_model", "text-embedding-3-small")

    oa = OpenAI(api_key=api_key)

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    col_sum = chroma.get_or_create_collection(name="paper_summaries") if write_to_chroma else None

    pdf_files = list(papers_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found under {papers_dir.resolve()}")
        return

    def embed_texts(texts: List[str]) -> List[List[float]]:
        resp = oa.embeddings.create(model=embed_model, input=texts)
        return [d.embedding for d in resp.data]

    def should_regenerate_from_cached(cached: dict) -> bool:
        """
        Decide whether to call the LLM again.
        """
        if not cached:
            return True

        # If the previous run explicitly marked a parse error
        if cached.get("_parse_error"):
            return True

        # If abstract looks like it contains JSON
        if abstract_looks_like_json(cached.get("summary_abstract", "")):
            return True

        # If it’s missing key fields
        if "summary_abstract" not in cached:
            return True

        # If legacy key exists but findings missing, we can fix without regen; treat as not regen
        return False

    repaired = 0
    regenerated = 0
    reused = 0

    for pdf_path in pdf_files:
        rel = pdf_path.relative_to(papers_dir)
        rel_norm = str(rel).replace("\\", "/")
        print(f"\nSummarizing PDF: {rel_norm}")

        fp = file_fingerprint_for_summary(pdf_path)
        cache_path = summaries_dir / short_cache_name(rel_norm)

        cached = read_cached_summary(cache_path, fp)

        # Determine mode for this file
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

        # If we can reuse cached, we still might "repair" schema & ensure chroma has markdown
        if not regen and cached is not None:
            summary_obj = dict(cached)
            core_norm = normalize_core(summary_obj)

            # If abstract was jsony but not flagged, we treat it as a "repair target"
            if abstract_looks_like_json(core_norm.get("summary_abstract", "")) and repair_bad_cached_only:
                # Force regen despite cache
                regen = True
            else:
                # Update cache if we changed schema (e.g., main_findings -> findings)
                if core_norm != summary_obj:
                    write_cached_summary(cache_path, fp, core_norm)
                    repaired += 1

                # Upsert markdown to Chroma (even when not regenerating)
                if write_to_chroma and col_sum is not None:
                    title = (core_norm.get("title") or "").strip()
                    authors = (core_norm.get("authors") or "").strip()
                    year = (core_norm.get("year") or "").strip()
                    doi = (core_norm.get("doi") or "").strip()

                    abstract = (core_norm.get("summary_abstract") or "").strip()
                    doc_for_chroma = abstract
                    embed_bits = []
                    if title: embed_bits.append(f"Title: {title}")
                    if authors: embed_bits.append(f"Authors: {authors}")
                    if year: embed_bits.append(f"Year: {year}")
                    if doi: embed_bits.append(f"DOI: {doi}")
                    if abstract: embed_bits.append(abstract)
                    embed_input = "\n".join(embed_bits).strip()

                    if embed_input:
                        emb = embed_texts([embed_input])[0]
                        sid = f"{rel_norm}::summary"
                        col_sum.delete(where={"source_file": rel_norm})
                        col_sum.upsert(
                            ids=[sid],
                            embeddings=[emb],
                            documents=[doc_for_chroma],
                            metadatas=[{
                                "unit_type": "paper",
                                "source_file": rel_norm,
                                "title": title,
                                "authors": authors,
                                "year": year,
                                "doi": doi,
                                "meta_source": core_norm.get("meta_source", ""),
                            }],
                        )
                reused += 1
                print("  Reused cached summary (and ensured markdown in Chroma).")
                continue

        # ---- Regenerate path ----
        meta = extract_title_authors_from_first_page(
            pdf_path,
            cache_dir=pdf_meta_cache_dir,
            doi_lookup=True,
        )
        title = (meta.get("title") or "").strip()
        authors = (meta.get("authors") or "").strip()
        year = (meta.get("year") or "").strip()
        doi = (meta.get("doi") or "").strip()
        meta_source = (meta.get("meta_source") or "").strip()

        full_text = build_full_text_from_pdf(pdf_path)
        if len(full_text) < 800:
            print("  Skipping (very little extractable text).")
            summary_obj = normalize_core({
                "unit_type": "paper",
                "source_file": rel_norm,
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "meta_source": meta_source,
                "summary_abstract": "",
                "key_contributions": [],
                "methods": [],
                "findings": [],
                "limitations": [],
                "keywords": [],
                "note": "no_text_or_scanned",
            })
            write_cached_summary(cache_path, fp, summary_obj)
            continue

        core = summarize_text_json(
            oa,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            text=full_text,
            model=gen_model,
        )

        summary_obj = normalize_core({
            "unit_type": "paper",
            "source_file": rel_norm,
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "meta_source": meta_source,
            **core,
        })

        write_cached_summary(cache_path, fp, summary_obj)
        regenerated += 1
        print("  Wrote summary JSON.")

        if write_to_chroma and col_sum is not None:
            abstract = (summary_obj.get("summary_abstract") or "").strip()
            doc_for_chroma = abstract
            embed_bits = []
            if title: embed_bits.append(f"Title: {title}")
            if authors: embed_bits.append(f"Authors: {authors}")
            if year: embed_bits.append(f"Year: {year}")
            if doi: embed_bits.append(f"DOI: {doi}")
            if abstract: embed_bits.append(abstract)
            embed_input = "\n".join(embed_bits).strip()

            if embed_input:
                emb = embed_texts([embed_input])[0]
                sid = f"{rel_norm}::summary"
                col_sum.delete(where={"source_file": rel_norm})
                col_sum.upsert(
                    ids=[sid],
                    embeddings=[emb],
                    documents=[doc_for_chroma],  # markdown for clean UI rendering
                    metadatas=[{
                        "unit_type": "paper",
                        "source_file": rel_norm,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "doi": doi,
                        "meta_source": meta_source,
                    }],
                )
                print("  Upserted markdown summary into Chroma collection 'paper_summaries'.")

    print(
        f"\nDone.\n"
        f"- reused cached: {reused}\n"
        f"- regenerated: {regenerated}\n"
        f"- repaired cached schema (no LLM): {repaired}\n"
        f"Output cache dir: {summaries_dir}"
    )


if __name__ == "__main__":
    main()