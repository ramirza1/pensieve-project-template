from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import json
import time
import hashlib
import urllib.request
import urllib.error
import os

import fitz  # PyMuPDF

CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "").strip()

# -------------------------
# Page text extraction
# -------------------------

def extract_pdf_pages(pdf_path: Path, min_chars_per_page: int = 30) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF.
    Returns a list of dicts: {page_num, text, char_count, has_text}
    """
    doc = fitz.open(str(pdf_path))
    pages: List[Dict[str, Any]] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = (page.get_text("text") or "").strip()
        pages.append(
            {
                "page_num": i + 1,  # 1-indexed
                "text": text,
                "char_count": len(text),
                "has_text": len(text) >= min_chars_per_page,
            }
        )

    doc.close()
    return pages


def extract_pdf_metadata(pdf_path: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    md = doc.metadata or {}
    doc.close()

    title = (md.get("title") or "").strip()
    author = (md.get("author") or "").strip()

    # Clean obvious junk
    if title.lower() in {"", "untitled"}:
        title = ""
    if author.lower() in {"", "unknown"}:
        author = ""

    return {"title": title, "authors": author}


def cut_after_references(pages: list[dict], min_hits: int = 2) -> list[dict]:
    """
    Cut the PDF text at the start of the References/Bibliography section.
    Only cut if we detect reference markers on at least `min_hits` pages.
    """
    markers = {"references", "bibliography", "reference list", "works cited"}
    hit_pages: list[int] = []

    for idx, p in enumerate(pages):
        t = (p.get("text") or "").lower()
        head = t[:2000]
        if any(m in head for m in markers):
            hit_pages.append(idx)

    if len(hit_pages) >= min_hits:
        return pages[:hit_pages[0]]

    return pages


# -------------------------
# DOI-first enrichment (Crossref)
# -------------------------

# Reasonably permissive DOI regex; we normalize after match.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)

def normalize_doi(raw: str) -> str:
    doi = (raw or "").strip()
    doi = doi.replace("doi:", "").replace("DOI:", "").strip()
    doi = doi.rstrip(".,);:]\"'>")
    return doi.lower()


def extract_doi_from_pages_text(pages: List[Dict[str, Any]], max_pages: int = 3) -> Optional[str]:
    """
    Look for a DOI in the first `max_pages` pages (by extracted text).
    Returns normalized DOI or None.
    """
    for p in pages[:max_pages]:
        txt = p.get("text") or ""
        m = _DOI_RE.search(txt)
        if m:
            doi = normalize_doi(m.group(0))
            if doi.startswith("10."):
                return doi
    return None


def pdf_fingerprint(pdf_path: Path) -> str:
    """
    Fast, stable-enough fingerprint to invalidate cache when file changes.
    """
    st = pdf_path.stat()
    payload = f"{pdf_path.name}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_path_for_pdf(pdf_path: Path, cache_dir: Path) -> Path:
    """
    One cache file per PDF, keyed on filename (plus a fingerprint inside).
    If you expect duplicate filenames, switch to hashing full path.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.name)
    return cache_dir / f"{safe}.json"


def load_cached_meta(pdf_path: Path, cache_dir: Path) -> Optional[dict]:
    cp = cache_path_for_pdf(pdf_path, cache_dir)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return None

    if data.get("fingerprint") != pdf_fingerprint(pdf_path):
        return None

    return data.get("meta")


def save_cached_meta(pdf_path: Path, cache_dir: Path, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": pdf_fingerprint(pdf_path),
        "saved_at": int(time.time()),
        "meta": meta,
    }
    cp = cache_path_for_pdf(pdf_path, cache_dir)
    cp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def crossref_user_agent() -> str:
    if CROSSREF_EMAIL:
        return f"PensievePDFIndexer/1.0 (mailto:{CROSSREF_EMAIL})"
    return "PensievePDFIndexer/1.0"

def fetch_crossref_metadata(doi: str, timeout_sec: int = 10) -> Optional[dict]:
    """
    Online: DOI -> Crossref canonical metadata.
    Returns dict {title, authors, year, doi, meta_source} or None.
    """
    doi = normalize_doi(doi)
    url = f"https://api.crossref.org/works/{doi}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": crossref_user_agent(),
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None

    msg = (data or {}).get("message") or {}
    titles = msg.get("title") or []
    title = (titles[0] if titles else "") or ""

    authors_list = msg.get("author") or []
    authors = []
    for a in authors_list:
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        name = " ".join([x for x in [given, family] if x]).strip()
        if name:
            authors.append(name)
    authors_str = ", ".join(authors)

    year = ""
    for key in ("published-print", "published-online", "issued", "created"):
        date_parts = ((msg.get(key) or {}).get("date-parts") or [])
        if date_parts and isinstance(date_parts, list) and date_parts[0]:
            try:
                year = str(date_parts[0][0])
                break
            except Exception:
                pass

    if not title:
        return None

    return {
        "title": title.strip(),
        "authors": authors_str.strip(),
        "year": year.strip(),
        "doi": doi,
        "meta_source": "crossref",
    }

# -------------------------
# Existing: layout-based parsing fallback
# -------------------------

_BAD_TITLE_PAT = re.compile(
    r"^(abstract|introduction|contents|table of contents|keywords|acknowledg|references)\b",
    re.IGNORECASE,
)

def _clean_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_title(line: str) -> bool:
    if not line or len(line) < 6:
        return False
    if _BAD_TITLE_PAT.search(line):
        return False
    if len(line) <= 3:
        return False
    return True


def extract_title_authors_from_first_page(
    pdf_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    doi_lookup: bool = True,
) -> dict:
    """
    Heuristic title/authors extraction using page 1 font sizes + positions.

    Priority:
      0) cache
      1) DOI -> Crossref
      2) embedded PDF metadata
      3) first-page layout heuristic
      4) filename

    Returns dict at least containing:
      {title, authors, meta_source}
    May also include:
      {doi, year}
    """
    # 0) Cache always wins
    if cache_dir is not None:
        cached = load_cached_meta(pdf_path, cache_dir)
        if cached:
            return cached

    # Filename fallback prepared early (used for any failure)
    fallback = pdf_path.stem.replace("_", " ").replace("-", " ").strip()

    # 1) DOI -> Crossref (gold standard)
    if doi_lookup and cache_dir is not None:
        try:
            pages = extract_pdf_pages(pdf_path, min_chars_per_page=1)
        except Exception:
            pages = []

        doi = extract_doi_from_pages_text(pages, max_pages=3) if pages else None
        if doi:
            meta = fetch_crossref_metadata(doi)
            if meta:
                save_cached_meta(pdf_path, cache_dir, meta)
                return {
                    "title": meta["title"],
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", ""),
                    "doi": meta.get("doi", ""),
                    "meta_source": "crossref",
                }

    # 2) Embedded PDF metadata (fallback)
    md = extract_pdf_metadata(pdf_path)
    title_md = (md.get("title") or "").strip()
    authors_md = (md.get("authors") or "").strip()
    if title_md:
        return {"title": title_md, "authors": authors_md, "meta_source": "pdf_meta"}

    # 3) Layout parse first page (guarded)
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return {"title": fallback, "authors": "", "meta_source": "filename"}

    try:
        if doc.page_count == 0:
            doc.close()
            return {"title": fallback, "authors": "", "meta_source": "filename"}

        try:
            page = doc.load_page(0)
        except Exception:
            doc.close()
            return {"title": fallback, "authors": "", "meta_source": "filename"}

        try:
            page_h = float(page.rect.height)
        except Exception:
            doc.close()
            return {"title": fallback, "authors": "", "meta_source": "filename"}

        try:
            d = page.get_text("dict")
        except Exception:
            doc.close()
            return {"title": fallback, "authors": "", "meta_source": "filename"}

        doc.close()

    except Exception:
        try:
            doc.close()
        except Exception:
            pass
        return {"title": fallback, "authors": "", "meta_source": "filename"}

    lines = []  # (y, size, text)
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join((sp.get("text") or "") for sp in spans)
            text = _clean_line(text)
            if not text:
                continue
            y = line.get("bbox", [0, 0, 0, 0])[1]
            max_size = max((sp.get("size") or 0) for sp in spans)
            lines.append((y, float(max_size), text))

    if not lines:
        return {"title": fallback, "authors": "", "meta_source": "filename"}

    top_cut = page_h * 0.35
    top_lines = [(y, s, t) for (y, s, t) in lines if y <= top_cut]
    if not top_lines:
        top_lines = lines[:]

    max_font = max(s for (_, s, _) in top_lines)

    title_cands = [(y, s, t) for (y, s, t) in top_lines if s >= 0.90 * max_font and _looks_like_title(t)]
    if not title_cands:
        title_cands = [(y, s, t) for (y, s, t) in top_lines if s >= 0.82 * max_font and _looks_like_title(t)]

    title_cands.sort(key=lambda x: x[0])

    title = ""
    if title_cands:
        title_parts = []
        first_y, _, _ = title_cands[0]
        for (y, _, t) in title_cands:
            if abs(y - first_y) <= 120:
                title_parts.append(t)
        title = _clean_line(" ".join(title_parts))

    authors = ""
    if title:
        title_y = title_cands[-1][0] if title_cands else None
        after = [(y, s, t) for (y, s, t) in top_lines if title_y is not None and y > title_y]
        after.sort(key=lambda x: x[0])

        author_lines = []
        for (_, _, t) in after[:10]:
            if _BAD_TITLE_PAT.search(t):
                break
            if re.search(r"\b(university|department|school|email|correspond)\b", t, re.IGNORECASE):
                continue
            if len(t) <= 2:
                continue
            author_lines.append(t)
            if len(author_lines) >= 2:
                break
        authors = _clean_line(" ".join(author_lines))

    if title:
        return {"title": title, "authors": authors, "meta_source": "first_page"}

    return {"title": fallback, "authors": "", "meta_source": "filename"}