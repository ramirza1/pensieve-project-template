from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict
import re

from docx import Document

HEADING_RE = re.compile(r"^Heading\s+(\d+)$", re.IGNORECASE)


@dataclass
class DocxBlock:
    heading_path: List[str]
    heading_levels: List[int]   # parallel list of heading levels, e.g. [2,3,4]
    text: str
    style: str
    para_index: int


def _is_heading_style(style_name: str) -> Optional[int]:
    """Return heading level if style is 'Heading N', else None."""
    if not style_name:
        return None
    m = HEADING_RE.match(style_name.strip())
    if not m:
        return None
    return int(m.group(1))


def extract_docx_blocks(docx_path: Path) -> List[DocxBlock]:
    """
    Extract paragraphs from a .docx and track current heading hierarchy.
    Returns one DocxBlock per non-empty paragraph.
    """
    doc = Document(str(docx_path))

    # stack entries are (level:int, heading_text:str)
    heading_stack: List[tuple[int, str]] = []
    blocks: List[DocxBlock] = []

    for i, p in enumerate(doc.paragraphs):
        text = (p.text or "").strip()
        style = (p.style.name if p.style is not None else "") or ""

        # Update heading stack when we hit a heading paragraph
        level = _is_heading_style(style)
        if level is not None:
            if text:  # skip empty headings
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
            continue  # headings don't become content blocks

        if not text:
            continue

        blocks.append(
            DocxBlock(
                heading_path=[h for (_, h) in heading_stack],
                heading_levels=[lvl for (lvl, _) in heading_stack],
                text=text,
                style=style,
                para_index=i,
            )
        )

    return blocks


def summarize_docx(docx_path: Path, max_preview_blocks: int = 12) -> None:
    blocks = extract_docx_blocks(docx_path)
    print(f"\nFILE: {docx_path.name}")
    print(f"Blocks extracted (non-empty paragraphs): {len(blocks)}")

    # count by top-level heading
    top_counts: Dict[str, int] = {}
    for b in blocks:
        top = b.heading_path[0] if b.heading_path else "(no heading)"
        top_counts[top] = top_counts.get(top, 0) + 1

    print("\nTop-level headings (first 10):")
    for k in list(top_counts.keys())[:10]:
        print(f"  - {k}: {top_counts[k]} blocks")

    print("\nPreview blocks:")
    for b in blocks[:max_preview_blocks]:
        heading = " > ".join(b.heading_path) if b.heading_path else "(no heading)"
        levels = " > ".join(str(x) for x in b.heading_levels) if b.heading_levels else "(no levels)"
        preview = b.text[:180].replace("\n", " ")
        print(f"\n[{b.para_index}] {heading}")
        print(f"Levels: {levels}")
        print(f"{preview}{'…' if len(b.text) > 180 else ''}")


def iter_docx_files(notes_root: Path) -> Iterable[Path]:
    for ext in ("*.docx", "*.DOCX"):
        yield from notes_root.rglob(ext)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract headings + paragraph blocks from .docx notes.")
    parser.add_argument("--notes_dir", type=str, default="data/inbox/notes", help="Root folder for notes")
    parser.add_argument("--file", type=str, default="", help="Optional: path to a specific .docx file")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        summarize_docx(path)
    else:
        notes_root = Path(args.notes_dir)
        files = list(iter_docx_files(notes_root))
        if not files:
            print(f"No .docx files found under: {notes_root.resolve()}")
        for f in files[:20]:
            summarize_docx(f)