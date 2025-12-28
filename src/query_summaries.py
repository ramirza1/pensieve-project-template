# src/query_summaries.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from openai import OpenAI


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_evidence_packets(
    docs: List[str],
    metas: List[Dict[str, Any]],
    max_chars_each: int = 1400,
) -> List[Dict[str, str]]:
    """
    Turns retrieved chunks into compact evidence packets for the model,
    with human-readable refs (page/para/heading).
    """
    packets = []
    for t, m in zip(docs, metas):
        text = (t or "").strip()
        if not text:
            continue

        # Build a reference string the UI can also show
        if m.get("source_type") == "pdf":
            ref = f"{m.get('source_file','')} p.{m.get('page_num','?')}"
        else:
            # notes
            hp = m.get("heading_path") or ""
            sp = m.get("start_para_index")
            ep = m.get("end_para_index")
            para = f"para {sp}-{ep}" if sp is not None and ep is not None else ""
            ref = f"{m.get('source_file','')} | {hp} | {para}".strip(" |")

        packets.append(
            {
                "ref": ref,
                "text": text[:max_chars_each],
            }
        )
    return packets


def summarize_doc_for_query(
    client: OpenAI,
    query: str,
    doc_meta: Dict[str, Any],
    evidence: List[Dict[str, str]],
    model: str = "gpt-4.1-mini",
) -> str:
    """
    Produces a grounded, query-focused summary for ONE doc using provided excerpts only.
    Returns markdown.
    """
    title = doc_meta.get("title") or doc_meta.get("paper_title") or doc_meta.get("doc_anchor") or ""
    authors = doc_meta.get("authors") or doc_meta.get("paper_authors") or ""
    year = doc_meta.get("year") or doc_meta.get("paper_year") or ""

    header_bits = [b for b in [title, authors, year] if b]
    doc_label = " — ".join(header_bits) if header_bits else (doc_meta.get("source_file") or "Document")

    # Compact evidence block
    evidence_lines = []
    for i, ev in enumerate(evidence, start=1):
        evidence_lines.append(f"[E{i}] {ev['ref']}\n{ev['text']}")
    evidence_block = "\n\n".join(evidence_lines)

    system = (
        "You are a careful research assistant. "
        "You must ONLY use the provided excerpts as evidence. "
        "If the excerpts do not contain enough info, say so plainly."
    )

    user = f"""Query: {query}
Document: {doc_label}

Task:
Write a concise, query-focused summary of what THIS document says about the query.

Output format (markdown):
- 3–6 bullets of key points specifically about the query
- If there are disagreements/nuances, note them
- End with: "Evidence used:" and list the excerpt ids you relied on (e.g., E1, E3)

Rules:
- Do NOT invent facts not present in excerpts.
- Prefer concrete claims over vague generalities.
- If the doc barely mentions the query, say "Limited coverage" and explain.

EXCERPTS:
{evidence_block}
"""

    # Prefer Responses API (newer OpenAI python), but keep it simple.
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.output_text.strip()
    except Exception:
        # Fallback for older setups
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (chat.choices[0].message.content or "").strip()


def cache_key(query: str, doc_key: str, evidence: List[Dict[str, str]]) -> str:
    """
    Makes a stable cache key from query + doc + evidence hashes.
    This means if the evidence changes (e.g., reindex), we regenerate.
    """
    ev_concat = "\n\n".join([e["ref"] + "\n" + e["text"] for e in evidence])
    return f"{_hash_text(query)}::{_hash_text(doc_key)}::{_hash_text(ev_concat)}"