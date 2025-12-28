# src/chunking.py
from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple


def _section_key(block: Dict[str, Any], anchor_level: int = 3) -> Tuple[str, ...]:
    """
    Returns a tuple key representing the section path up to anchor_level.
    Example: [H2 theme, H3 paper] if available.
    Falls back gracefully if headings are missing.
    """
    hp = block.get("heading_path") or []
    hl = block.get("heading_levels") or []

    # keep headings whose level <= anchor_level
    out = []
    for text, lvl in zip(hp, hl):
        if lvl <= anchor_level:
            out.append(text)
    return tuple(out) if out else ("(no heading)",)


def chunk_blocks(
    blocks: List[Dict[str, Any]],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    anchor_level: int = 3,
) -> List[Dict[str, Any]]:
    """
    Build chunks within heading-anchored sections (e.g., per-paper).
    Handles bullet lists by keeping them together when possible.
    """
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: b.get("para_index", 0))

    # Group into sections
    sections: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for b in blocks:
        key = _section_key(b, anchor_level=anchor_level)
        sections.setdefault(key, []).append(b)

    chunks: List[Dict[str, Any]] = []

    for key, sec_blocks in sections.items():
        items = []
        for b in sec_blocks:
            t = (b.get("text") or "").strip()
            if not t:
                continue
            items.append((b["para_index"], t, b.get("heading_path") or [], b.get("heading_levels") or []))

        if not items:
            continue

        section_heading_path = items[0][2]
        section_heading_levels = items[0][3]

        buf = ""
        buf_start = items[0][0]
        buf_end = items[0][0]

        def flush():
            nonlocal buf, buf_start, buf_end
            if buf.strip():
                chunks.append({
                    "text": buf.strip(),
                    "heading_path": section_heading_path,
                    "heading_levels": section_heading_levels,
                    "start_para_index": buf_start,
                    "end_para_index": buf_end,
                })
            buf = ""

        for i, (para_index, t, _, _) in enumerate(items):
            # More sophisticated bullet detection
            is_bullet = bool(re.match(r'^[\s]*[\-\*•○▪►]|\d+\.', t.lstrip()))
            
            # Check if next item is also a bullet (keep them together)
            next_is_bullet = False
            if i + 1 < len(items):
                next_text = items[i + 1][1]
                next_is_bullet = bool(re.match(r'^[\s]*[\-\*•○▪►]|\d+\.', next_text.lstrip()))
            
            # For bullet-heavy content, be more aggressive about keeping bullets together
            if buf and (len(buf) + 2 + len(t)) > chunk_size:
                # Only flush if we've accumulated a reasonable chunk AND we're not mid-bullet-list
                if len(buf) > chunk_size * 0.5 and not (is_bullet and next_is_bullet):
                    flush()
                    tail = chunks[-1]["text"][-chunk_overlap:] if chunks else ""
                    buf = tail + "\n" + t if tail else t
                    buf_start = para_index
                    buf_end = para_index
                else:
                    # Keep building the bullet list
                    buf += "\n" + t
                    buf_end = para_index
            else:
                # Normal addition - use single newline for bullets
                separator = "\n" if is_bullet else "\n\n"
                if buf:
                    buf += separator + t
                else:
                    buf = t
                    buf_start = para_index
                buf_end = para_index

        flush()

    return chunks
