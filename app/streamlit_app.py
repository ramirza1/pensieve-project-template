"""
Pensieve Streamlit App
======================

A semantic search interface for your research notes and papers.

Deployment modes:
- Local: Uses data/_local/chroma_db (no B2 credentials needed)
- Cloud: Downloads from Backblaze B2 into data/_server/chroma_db

The app auto-detects which mode to use based on whether B2 credentials are present.
You can override this by setting DEPLOYMENT_MODE=local or DEPLOYMENT_MODE=b2 in .env
"""

from __future__ import annotations

import os
import re
import math
import hashlib
import shutil
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

import yaml
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import chromadb


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def load_config() -> dict:
    """Load configuration from config.yaml."""
    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_secret_or_env(key: str, default: str | None = None) -> str | None:
    """Get a value from Streamlit secrets (cloud) or environment variables (local)."""
    try:
        v = st.secrets.get(key, None)
        if v is not None:
            return str(v)
    except Exception:
        pass
    v = os.getenv(key, None)
    return v if v is not None else default


def _truthy(v: str | None) -> bool:
    """Check if a string value represents 'true'."""
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).resolve().parent.parent


def _abs_in_repo(rel: str) -> Path:
    """Convert a relative path to absolute within the repo."""
    return _repo_root() / rel


# ============================================================================
# DEPLOYMENT MODE DETECTION
# ============================================================================

def get_deployment_mode() -> str:
    """
    Determine deployment mode:
    - "local": Use local ChromaDB folder (no download)
    - "b2": Download from Backblaze B2
    - "auto": Auto-detect based on credentials (default)
    
    Set DEPLOYMENT_MODE in .env or config.yaml to override.
    """
    cfg = load_config()
    mode = (os.getenv("DEPLOYMENT_MODE") or cfg.get("deployment_mode") or "auto").strip().lower()
    
    if mode not in ("local", "b2", "auto"):
        mode = "auto"

    if mode == "auto":
        # Auto-detect: use B2 if credentials exist, otherwise local
        b2_key = _get_secret_or_env("B2_KEY_ID") or _get_secret_or_env("B2_APPLICATION_KEY_ID")
        b2_secret = _get_secret_or_env("B2_APP_KEY") or _get_secret_or_env("B2_APPLICATION_KEY")
        has_b2 = bool(b2_key) and bool(b2_secret)
        return "b2" if has_b2 else "local"

    return mode


# ============================================================================
# CHROMA DB INITIALIZATION
# ============================================================================

@st.cache_resource(show_spinner=False)
def ensure_chroma_db_available() -> Path:
    """
    Ensure ChromaDB folder exists and is populated.
    
    - Local mode: Uses data/_local/chroma_db (expects you ran indexing locally)
    - B2 mode: Downloads into data/_server/chroma_db from Backblaze B2
    """
    cfg = load_config()
    paths = cfg.get("paths") or {}

    # Get paths from config (with defaults)
    local_dir = str(paths.get("chroma_dir_local", "data/_local/chroma_db"))
    server_dir = str(paths.get("chroma_dir_server", "data/_server/chroma_db"))

    mode = get_deployment_mode()
    force_download = _truthy(_get_secret_or_env("FORCE_B2_DOWNLOAD", "false"))

    # Choose target directory based on mode
    chroma_dir = _abs_in_repo(server_dir if mode == "b2" else local_dir)

    # Local mode: just ensure directory exists
    if mode != "b2":
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return chroma_dir

    # B2 mode: download if missing/empty or forced
    if (not force_download) and chroma_dir.exists() and any(chroma_dir.iterdir()):
        return chroma_dir

    # Get B2 credentials
    b2_key_id = _get_secret_or_env("B2_KEY_ID") or _get_secret_or_env("B2_APPLICATION_KEY_ID")
    b2_app_key = _get_secret_or_env("B2_APP_KEY") or _get_secret_or_env("B2_APPLICATION_KEY")
    b2_bucket = _get_secret_or_env("B2_BUCKET_NAME", "pensieve-db")
    b2_prefix = _get_secret_or_env("B2_PREFIX", "chroma_db/")

    if not b2_key_id or not b2_app_key:
        st.error("B2 credentials not configured (need B2_KEY_ID and B2_APP_KEY).")
        st.info("For local development, run the indexing pipeline first: `python scripts/update_and_deploy.py --skip-upload`")
        st.stop()

    # Download from B2
    ph = st.empty()
    ph.info("🧠 Loading Pensieve database from B2 (first load may take a minute)…")

    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo

        # Normalize prefix
        b2_prefix = (b2_prefix or "").strip()
        if b2_prefix and not b2_prefix.endswith("/"):
            b2_prefix += "/"

        # Wipe and recreate target folder
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        # Connect to B2
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", b2_key_id, b2_app_key)
        bucket = b2_api.get_bucket_by_name(b2_bucket)

        # Download all files
        file_count = 0
        for file_version, _ in bucket.ls(folder_to_list=b2_prefix, recursive=True):
            file_name = file_version.file_name
            if not file_name.startswith(b2_prefix):
                continue

            local_rel = file_name[len(b2_prefix):]
            if not local_rel:
                continue

            local_path = chroma_dir / local_rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            bucket.download_file_by_name(file_name).save_to(str(local_path))
            file_count += 1

        if not any(chroma_dir.iterdir()):
            ph.error(f"Downloaded 0 files; DB folder is empty: {chroma_dir}")
            st.stop()

        ph.success(f"✅ Database loaded ({file_count} files)")
        return chroma_dir

    except Exception as e:
        ph.error(f"Failed to download database from B2: {e}")
        st.stop()


# ============================================================================
# HYBRID SEARCH MODULE
# ============================================================================

class BM25:
    """BM25 keyword matching algorithm for hybrid search."""
    
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1
        self.n_docs = len(documents)
        
        self.doc_freqs: dict[str, int] = Counter()
        self.term_freqs: list[dict[str, int]] = []
        
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1
    
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\b[a-z0-9]+\b', text.lower())
    
    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = self._tokenize(query)
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
            score += idf * (numerator / denominator)
        
        return score


@dataclass
class QueryAnalysis:
    """Analysis of search query to determine optimal search strategy."""
    original_query: str
    is_likely_author_search: bool = False
    is_likely_exact_phrase: bool = False
    is_likely_concept_search: bool = False


def analyze_query(query: str) -> QueryAnalysis:
    """Detect query type to adjust search weights."""
    analysis = QueryAnalysis(original_query=query)
    query_terms = re.findall(r'\b[a-z0-9]+\b', query.lower())
    
    concept_indicators = {
        'theory', 'effect', 'model', 'analysis', 'research', 'study',
        'reasoning', 'bias', 'cognition', 'political', 'social', 'media',
        'communication', 'psychology', 'behavior', 'perception', 'attitude',
        'motivated', 'selective', 'exposure', 'processing', 'information',
        'deliberation', 'discourse', 'polarization', 'persuasion'
    }
    
    words = query.split()
    
    if 1 <= len(words) <= 4:
        all_capitalized = all(w[0].isupper() for w in words if w)
        no_concept_words = not any(w.lower() in concept_indicators for w in words)
        if all_capitalized and no_concept_words:
            analysis.is_likely_author_search = True
    
    if '"' in query or "'" in query:
        analysis.is_likely_exact_phrase = True
    
    if any(term in concept_indicators for term in query_terms):
        analysis.is_likely_concept_search = True
    
    return analysis


def _normalize_text(text: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()


def author_match_score(query: str, authors_field: str) -> float:
    if not authors_field:
        return 0.0
    
    query_norm = _normalize_text(query)
    authors_norm = _normalize_text(authors_field)
    
    if query_norm in authors_norm:
        return 1.0
    
    query_parts = query_norm.split()
    authors_parts = authors_norm.split()
    
    matched_parts = sum(1 for qp in query_parts if any(qp in ap or ap in qp for ap in authors_parts))
    if query_parts and matched_parts > 0:
        return matched_parts / len(query_parts)
    
    return 0.0


def title_match_score(query: str, title_field: str) -> float:
    if not title_field:
        return 0.0
    
    query_norm = _normalize_text(query)
    title_norm = _normalize_text(title_field)
    
    if query_norm in title_norm:
        return 1.0
    
    query_words = set(query_norm.split())
    title_words = set(title_norm.split())
    
    if not query_words:
        return 0.0
    
    overlap = query_words & title_words
    return len(overlap) / len(query_words)


def hybrid_search(
    collection,
    query_embedding: list[float],
    query: str,
    top_k: int = 10,
    initial_k: int = 50,
) -> dict:
    """
    Perform hybrid search combining semantic + BM25 + metadata matching.
    """
    analysis = analyze_query(query)
    
    # Adjust weights based on query type
    if analysis.is_likely_author_search:
        semantic_weight, bm25_weight, metadata_weight = 0.3, 0.2, 0.5
    elif analysis.is_likely_exact_phrase:
        semantic_weight, bm25_weight, metadata_weight = 0.3, 0.5, 0.2
    else:
        semantic_weight, bm25_weight, metadata_weight = 0.5, 0.3, 0.2
    
    author_boost = 2.0
    title_boost = 1.5
    exact_phrase_boost = 1.5
    
    try:
        fetch_k = min(initial_k, collection.count())
    except Exception:
        fetch_k = initial_k
    
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=['documents', 'metadatas', 'distances'],
    )
    
    sem_docs = semantic_results.get('documents', [[]])[0]
    sem_metas = semantic_results.get('metadatas', [[]])[0]
    sem_dists = semantic_results.get('distances', [[]])[0]
    
    if not sem_docs:
        return semantic_results
    
    bm25 = BM25(sem_docs)
    scored_results = []
    
    for i, (doc, meta, dist) in enumerate(zip(sem_docs, sem_metas, sem_dists)):
        meta = meta or {}
        
        semantic_score = 1.0 / (1.0 + dist)
        bm25_score = bm25.score(query, i)
        
        authors = meta.get('authors', '') or meta.get('paper_authors', '') or ''
        title = meta.get('title', '') or meta.get('paper_title', '') or ''
        
        author_score = author_match_score(query, authors)
        title_score = title_match_score(query, title)
        exact_match = 1.0 if query.lower() in doc.lower() else 0.0
        
        scored_results.append({
            'doc': doc, 'meta': meta, 'dist': dist,
            'semantic': semantic_score, 'bm25': bm25_score,
            'author': author_score, 'title': title_score, 'exact': exact_match,
        })
    
    # Normalize BM25 scores
    bm25_scores = [r['bm25'] for r in scored_results]
    if any(s > 0 for s in bm25_scores):
        min_s, max_s = min(bm25_scores), max(bm25_scores)
        if max_s > min_s:
            for r in scored_results:
                r['bm25'] = (r['bm25'] - min_s) / (max_s - min_s)
    
    # Compute final scores
    for r in scored_results:
        base_score = (
            semantic_weight * r['semantic'] +
            bm25_weight * r['bm25'] +
            metadata_weight * max(r['author'], r['title'])
        )
        
        boost = 1.0
        if r['author'] > 0.5:
            boost *= author_boost * (1.5 if analysis.is_likely_author_search else 1.0)
        if r['title'] > 0.3:
            boost *= title_boost
        if r['exact'] > 0:
            boost *= exact_phrase_boost
        
        r['final_score'] = base_score * boost
    
    scored_results.sort(key=lambda r: r['final_score'], reverse=True)
    top_results = scored_results[:top_k]
    
    return {
        'documents': [[r['doc'] for r in top_results]],
        'metadatas': [[r['meta'] for r in top_results]],
        'distances': [[r['dist'] for r in top_results]],
    }


# ============================================================================
# EMBEDDING & QUERY HELPERS
# ============================================================================

def embed_query(client: OpenAI, text: str, model: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def run_query(col, q_emb: list[float], query_text: str, top_k: int) -> dict:
    return hybrid_search(col, q_emb, query_text, top_k=top_k)


# ============================================================================
# THEME HELPERS
# ============================================================================

def _init_theme_from_query_params(default: str = "light") -> None:
    try:
        qp = st.query_params
        t = (qp.get("theme") or "").strip().lower()
        if t not in ("light", "dark"):
            t = default
    except Exception:
        t = default

    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = t


def _persist_theme_to_query_params(theme: str) -> None:
    try:
        st.query_params["theme"] = theme
    except Exception:
        pass


# ============================================================================
# AI SUMMARY HELPERS
# ============================================================================

def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def _doc_key_from_meta(meta: dict, label: str) -> str:
    if label == "NOTES":
        return (meta.get("doc_id") or meta.get("source_file") or "unknown_doc").strip()
    return (meta.get("source_file") or meta.get("doc_id") or "unknown_doc").strip()


def _doc_label_from_meta(meta: dict, label: str) -> str:
    if label == "PAPERS":
        title = (meta.get("title") or "").strip()
        authors = (meta.get("authors") or "").strip()
        year = (meta.get("year") or "").strip()
        bits = [b for b in [title, authors, year] if b]
        return " — ".join(bits) if bits else (meta.get("source_file") or "Paper")
    else:
        theme = (meta.get("theme") or "").strip()
        paper_title = (meta.get("paper_title") or meta.get("doc_anchor") or "").strip()
        paper_authors = (meta.get("paper_authors") or "").strip()
        paper_year = (meta.get("paper_year") or "").strip()
        bits = []
        if theme:
            bits.append(theme)
        p = paper_title
        if p:
            if paper_authors:
                p += f" — {paper_authors}"
            if paper_year:
                p += f" ({paper_year})"
            bits.append(p)
        return "  >  ".join(bits) if bits else (meta.get("source_file") or "Note")


def _display_header_from_meta(meta: dict, label: str) -> str:
    meta = meta or {}
    if label == "PAPERS":
        title = (meta.get("title") or "").strip()
        authors = (meta.get("authors") or "").strip()
        year = (meta.get("year") or "").strip()
    else:
        title = (meta.get("paper_title") or meta.get("doc_anchor") or "").strip()
        authors = (meta.get("paper_authors") or "").strip()
        year = (meta.get("paper_year") or "").strip()
    parts = [p for p in [title, authors, year] if p]
    return ", ".join(parts)


@st.cache_data(show_spinner=False)
def cached_query_focused_summary(
    cache_id: str,
    model: str,
    query: str,
    label: str,
    doc_label: str,
    doc_text: str,
) -> str:
    client = OpenAI()
    system = (
        "You are a careful research assistant. "
        "You must ONLY use the provided document text. "
        "If the document does not clearly address the query, say so."
    )
    user = f"""Query: {query}
Document: {doc_label}
Source type: {label}

Task:
Produce a query-focused summary of what THIS document says about the query.

Output (markdown):
- Start with a 1–2 sentence direct answer (or 'Limited coverage' if appropriate).
- Then 3–6 bullets of specific insights/claims tied to the query.
- If there are caveats/nuance/competing interpretations present, include them.
- End with: "Confidence: High/Medium/Low" based on how directly the text speaks to the query.

Rules:
- Do NOT invent details not in the text.
- Prefer abstract, high-level phrasing. Do not reproduce distinctive sentences or extended passages from the document.
- If the text is broad, explain what is and isn't supported.

DOCUMENT TEXT:
{doc_text}
"""
    try:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (chat.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error generating summary: {e}"


# ============================================================================
# UI HELPERS
# ============================================================================

def inject_css(mode: str) -> None:
    if mode == "dark":
        bg, panel = "#0f111a", "#141827"
        border = "rgba(255,255,255,0.10)"
        text, muted = "rgba(255,255,255,0.92)", "rgba(255,255,255,0.70)"
        link = "#7aa2ff"
        input_bg, pill_bg, hover_bg = "rgba(255,255,255,0.06)", "rgba(255,255,255,0.06)", "rgba(255,255,255,0.04)"
    else:
        bg, panel = "#fbf7ef", "#ffffff"
        border = "rgba(0,0,0,0.08)"
        text, muted = "rgba(0,0,0,0.88)", "rgba(0,0,0,0.62)"
        link = "#1f5fbf"
        input_bg, pill_bg, hover_bg = "rgba(0,0,0,0.03)", "rgba(0,0,0,0.03)", "rgba(0,0,0,0.03)"

    st.markdown(f"""
        <style>
        .block-container {{ max-width: 1400px; padding-left: 3rem; padding-right: 3rem; }}
        .stApp {{ background: {bg}; color: {text}; }}
        html, body, [class*="css"] {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif; }}
        a, a:visited {{ color: {link} !important; }}
        .muted {{ color: {muted}; }}
        .stExpander {{ border-radius: 12px; border: 1px solid {border}; background: {panel}; }}
        div[data-baseweb="input"] > div {{ background: {input_bg}; }}
        div[data-baseweb="select"] > div {{ background: {input_bg}; }}
        .stNumberInput, .stSelectbox {{ margin-top: -2px; }}
        div[data-testid="stExpander"] > details > summary {{ background-color: transparent !important; }}
        div[data-testid="stExpander"] summary > div {{ background-color: transparent !important; }}
        div[data-testid="stExpander"] summary:hover {{ background-color: {hover_bg} !important; }}
        .pensieve-meta {{ margin: 0 0 0.35rem 0; }}
        .pensieve-kv {{ margin: 0.15rem 0; line-height: 1.25; }}
        .pensieve-kv .k {{ font-weight: 600; opacity: 0.9; }}
        .pensieve-kv .v {{ opacity: 0.95; }}
        .pensieve-divider {{ height: 1px; background: {border}; margin: 0.55rem 0; }}
        div.stButton > button {{ background: {input_bg} !important; color: {text} !important; border: 1px solid {border} !important; border-radius: 10px !important; }}
        div.stButton > button:hover {{ background: {pill_bg} !important; }}
        </style>
    """, unsafe_allow_html=True)


def fetch_note_summaries_by_doc_ids(col_sum, doc_ids: list[str]) -> dict[str, str]:
    if col_sum is None or not doc_ids:
        return {}
    seen = set()
    doc_ids = [d for d in doc_ids if d and not (d in seen or seen.add(d))]
    sids = [f"{did}::summary" for did in doc_ids]
    got = col_sum.get(ids=sids, include=["documents", "metadatas"])
    out: dict[str, str] = {}
    for sid, doc, meta in zip(got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []):
        meta = meta or {}
        did = meta.get("doc_id") or (sid[:-len("::summary")] if sid.endswith("::summary") else sid)
        out[did] = (doc or "").strip()
    return out


def fetch_paper_summaries_by_source_files(col_sum, source_files: list[str]) -> dict[str, str]:
    if col_sum is None or not source_files:
        return {}
    seen = set()
    source_files = [s for s in source_files if s and not (s in seen or seen.add(s))]
    sids = [f"{sf}::summary" for sf in source_files]
    got = col_sum.get(ids=sids, include=["documents", "metadatas"])
    out: dict[str, str] = {}
    for sid, doc, meta in zip(got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []):
        meta = meta or {}
        sf = meta.get("source_file") or (sid[:-len("::summary")] if sid.endswith("::summary") else sid)
        out[sf] = (doc or "").strip()
    return out


def render_results(
    label: str,
    res: dict,
    query: str,
    ai_model: str,
    summaries: dict[str, str] | None = None,
    debug: bool = False,
    enable_ai_snippets: bool = False,
):
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    icon = "📝" if label == "NOTES" else "📄"
    st.subheader(f"{icon} {label}")

    for i, (doc_text, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        meta = meta or {}
        src = meta.get("source_file", "unknown")
        parts = src.split("/")
        filename = parts[-1] if parts else src
        folder = " / ".join(parts[:-1]) if len(parts) > 1 else ""
        
        source_type = (meta.get("source_type") or "").strip()
        chunk_ix = meta.get("chunk_index")
        heading_path = (meta.get("heading_path") or "").strip()
        
        theme = (meta.get("theme") or "").strip()
        paper_title = (meta.get("paper_title") or "").strip()
        paper_authors = (meta.get("paper_authors") or "").strip()
        paper_year = (meta.get("paper_year") or "").strip()
        
        pdf_title = (meta.get("title") or "").strip()
        pdf_authors = (meta.get("authors") or "").strip()
        
        page = meta.get("page_num")
        page_str = f"p.{page}" if page else ""

        header_main = _display_header_from_meta(meta, label)
        display = header_main if header_main else (f"{folder} / {filename}" if folder else filename)

        exp_title = f"[{i}] {display}"
        if page_str:
            exp_title += f" • {page_str}"
        if debug:
            exp_title += f" • chunk={chunk_ix} • d={dist:.3f}" if chunk_ix is not None else f" • d={dist:.3f}"

        key = (meta.get("doc_id") or "").strip() if label == "NOTES" else (meta.get("source_file") or "").strip()
        summary_text = (summaries or {}).get(key, "")

        def _kv(k: str, value: str) -> str:
            v = (value or "").strip()
            return f"<div class='pensieve-kv'><span class='k'>{k}:</span> <span class='v'>{v}</span></div>" if v else ""

        if source_type == "note":
            body_html = "<div class='pensieve-meta'>" + _kv("Theme", theme) + _kv("Title", paper_title) + _kv("Authors", paper_authors) + _kv("Year", paper_year) + "</div>"
        else:
            body_html = "<div class='pensieve-meta'>" + _kv("Title", pdf_title) + _kv("Authors", pdf_authors) + _kv("Year", (meta.get("year") or "").strip()) + "</div>"

        if body_html == "<div class='pensieve-meta'></div>":
            body_html = "<div class='pensieve-meta muted'>No metadata available.</div>"

        with st.expander(exp_title, expanded=False):
            st.markdown(body_html, unsafe_allow_html=True)
            st.markdown("<div class='pensieve-divider'></div>", unsafe_allow_html=True)

            if summary_text:
                st.markdown("**Summary**")
                st.markdown(summary_text)
            else:
                st.markdown("_No summary found for this item yet._")

            if enable_ai_snippets and query and doc_text:
                st.markdown("<div class='pensieve-divider'></div>", unsafe_allow_html=True)
                st.markdown("**Query-focused insights**")
                
                doc_key = _doc_key_from_meta(meta, label)
                doc_label = _doc_label_from_meta(meta, label)
                ck = f"{_hash_text(query)}::{_hash_text(doc_key)}::{_hash_text(doc_text)}::{ai_model}"
                btn_key = f"gen_qsum::{label}::{i}::{ck}"
                
                if st.button("Generate / refresh", key=btn_key, use_container_width=True):
                    st.session_state[f"show_qsum::{btn_key}"] = True
                    st.rerun()
                
                if st.session_state.get(f"show_qsum::{btn_key}", False):
                    with st.spinner("Summarizing…"):
                        snippet = cached_query_focused_summary(ck, ai_model, query, label, doc_label, doc_text[:18000])
                    st.markdown(snippet if snippet else "_No output returned._")

            # Metadata expander
            if (label == "PAPERS") or heading_path or debug:
                st.markdown("<div class='pensieve-divider'></div>", unsafe_allow_html=True)
                with st.expander("Metadata", expanded=False):
                    meta_html = "<div class='pensieve-meta'>" + _kv("Heading path", heading_path) + _kv("Source file", src) + _kv("Source type", source_type) + _kv("Doc ID", (meta.get("doc_id") or "").strip()) + "</div>"
                    st.markdown(meta_html, unsafe_allow_html=True)
                    if debug:
                        st.json(meta)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Pensieve", layout="wide")

    _init_theme_from_query_params()
    inject_css(st.session_state.get("theme_mode", "light"))

    load_dotenv()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY") or _get_secret_or_env("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to your .env file.")
        st.stop()

    cfg = load_config()
    repo = _repo_root()

    # Get ChromaDB (local or from B2)
    chroma_dir = ensure_chroma_db_available()

    # Config values
    top_k_default = int((cfg.get("retrieval") or {}).get("top_k", 8))
    embed_model = cfg.get("embedding_model", "text-embedding-3-small")
    ai_model = (cfg.get("summarization") or {}).get("model") or (cfg.get("app") or {}).get("query_snippet_model") or "gpt-4.1-mini"
    allow_debug = bool((cfg.get("app") or {}).get("allow_debug", False))
    github_url = (cfg.get("app") or {}).get("github_url", "").strip() or "https://github.com/yourusername/pensieve"

    # Initialize clients
    oa = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=str(chroma_dir))

    notes = chroma.get_or_create_collection(name="notes_docs")
    papers = chroma.get_or_create_collection(name="papers_docs")
    note_summaries = chroma.get_or_create_collection(name="note_summaries")
    paper_summaries = chroma.get_or_create_collection(name="paper_summaries")

    # -------------------- HEADER --------------------
    left, right = st.columns([2.1, 1.4], gap="large")

    with left:
        h1, toggles = st.columns([4, 1.25], vertical_alignment="center")
        with h1:
            st.markdown("<h1 style='margin: 0 0 0.15em 0; font-size: 2.35rem;'>🧠 Pensieve</h1>", unsafe_allow_html=True)
        with toggles:
            c1, c2 = st.columns([1, 1], vertical_alignment="center")
            with c1:
                if st.button("🌗", key="btn_theme", help="Toggle light/dark", use_container_width=True):
                    st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
                    _persist_theme_to_query_params(st.session_state.theme_mode)
                    st.rerun()
            with c2:
                debug = st.toggle("🐛", key="toggle_debug", value=False, help="Debug mode") if allow_debug else False

        st.markdown("""
            <div style="font-size: 1.18em; line-height: 1.5; margin-bottom: 0.55em;">
              A memory basin for <strong>papers</strong> and <strong>notes</strong>.
            </div>
            <div class="muted" style="font-size: 1.02em; line-height: 1.55; margin-bottom: 0.55em;">
              Just like Dumbledore stored memories in the Pensieve, this tool helps you retrieve knowledge 
              from your notes and readings—making your research instantly searchable and intelligently summarized.
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div class='muted' style='font-size: 0.98em;'>🔗 Code: <a href='{github_url}' target='_blank'>{github_url}</a></div>", unsafe_allow_html=True)

        if debug:
            st.caption(f"mode={get_deployment_mode()}  chroma_dir={chroma_dir}")

    with right:
        img_path = repo / "assets" / "pensieve.jpg"
        if img_path.exists():
            st.image(Image.open(img_path), use_container_width=True)

    # -------------------- CONTROLS --------------------
    c_search, c_notes, c_papers, c_k, c_ai = st.columns([2.6, 0.9, 0.9, 1.2, 1.4], vertical_alignment="bottom")

    with c_search:
        q = st.text_input("🔍 Search", placeholder='e.g., motivated reasoning, causal inference', label_visibility="collapsed", key="q")
    with c_notes:
        use_notes = st.checkbox("📝 Notes", value=True, key="use_notes")
    with c_papers:
        use_papers = st.checkbox("📄 Papers", value=True, key="use_papers")
    with c_k:
        top_k_ui = st.number_input("Number of results", min_value=1, max_value=25, value=top_k_default, step=1, key="top_k_ui")
    with c_ai:
        enable_ai_snippets = st.checkbox("✨ AI snippets", value=False, key="enable_ai_snippets", help="Generate query-focused insights per result")

    # -------------------- QUERY --------------------
    if q:
        q_emb = embed_query(oa, q, model=embed_model)

        if use_notes:
            res_notes = run_query(notes, q_emb, q, int(top_k_ui))
            note_doc_ids = [(m or {}).get("doc_id") for m in res_notes["metadatas"][0] if (m or {}).get("doc_id")]
            note_sum_map = fetch_note_summaries_by_doc_ids(note_summaries, note_doc_ids)
            render_results("NOTES", res_notes, query=q, ai_model=ai_model, summaries=note_sum_map, debug=debug, enable_ai_snippets=enable_ai_snippets)

        if use_papers:
            res_papers = run_query(papers, q_emb, q, int(top_k_ui))
            paper_source_files = [(m or {}).get("source_file") for m in res_papers["metadatas"][0] if (m or {}).get("source_file")]
            paper_sum_map = fetch_paper_summaries_by_source_files(paper_summaries, paper_source_files)
            render_results("PAPERS", res_papers, query=q, ai_model=ai_model, summaries=paper_sum_map, debug=debug, enable_ai_snippets=enable_ai_snippets)

    st.markdown("---")
    st.caption("Built with Pensieve · Fork this template on GitHub")


if __name__ == "__main__":
    main()