from __future__ import annotations

import os
import hashlib
import shutil
from pathlib import Path

import yaml
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import chromadb


# ============================================================
# Pensieve (Template) Streamlit App
# - Local mode: uses data/_local/chroma_db
# - Cloud mode: downloads DB from Backblaze B2 into data/_server/chroma_db
#
# B2 env/secrets expected (standardized):
#   B2_KEY_ID
#   B2_APP_KEY
#   B2_BUCKET_NAME (optional; default "pensieve-db")
#   B2_PREFIX (optional; default "chroma_db/")
#
# Optional:
#   FORCE_B2_DOWNLOAD="true"  (forces re-download on next load)
# ============================================================


# ---------- Config helpers ----------
def load_config() -> dict:
    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_secret_or_env(key: str, default: str | None = None) -> str | None:
    """
    Prefer Streamlit secrets (Cloud), fallback to env (local).
    """
    try:
        v = st.secrets.get(key, None)  # type: ignore[attr-defined]
        if v is not None:
            return str(v)
    except Exception:
        pass
    v = os.getenv(key, None)
    return v if v is not None else default


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "on")


def get_deployment_mode() -> str:
    """
    deployment_mode:
      - "local": never download; use local DB folder
      - "b2": always download from B2 into server folder
      - "auto": use B2 if credentials exist; else local
    """
    cfg = load_config()
    mode = (os.getenv("DEPLOYMENT_MODE") or cfg.get("deployment_mode") or "auto").strip().lower()
    if mode not in ("local", "b2", "auto"):
        mode = "auto"

    if mode == "auto":
        # auto-detect based on credentials
        has_b2 = bool(_get_secret_or_env("B2_KEY_ID")) and bool(_get_secret_or_env("B2_APP_KEY"))
        return "b2" if has_b2 else "local"

    return mode


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _abs_in_repo(rel: str) -> Path:
    return _repo_root() / rel


# ---------- B2 Download Helper ----------
@st.cache_resource(show_spinner=False)
def ensure_chroma_db_available() -> Path:
    """
    Ensures Chroma DB folder exists and is populated.
    - Local mode: uses data/_local/chroma_db (expects you built it locally)
    - B2 mode: downloads into data/_server/chroma_db
    """
    cfg = load_config()
    paths = (cfg.get("paths") or {})

    # Defaults (safe for template)
    local_dir = str(paths.get("chroma_dir_local", "data/_local/chroma_db"))
    server_dir = str(paths.get("chroma_dir_server", "data/_server/chroma_db"))

    mode = get_deployment_mode()

    # Force refresh toggle
    force_download = _truthy(_get_secret_or_env("FORCE_B2_DOWNLOAD", "false"))

    # Choose target dir by mode
    chroma_dir = _abs_in_repo(server_dir if mode == "b2" else local_dir)

    # Local mode: just ensure dir exists (no download)
    if mode != "b2":
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return chroma_dir

    # B2 mode: download if missing/empty or forced
    marker_file = chroma_dir / ".b2_downloaded"
    if (not force_download) and chroma_dir.exists() and any(chroma_dir.iterdir()):
        return chroma_dir

    b2_key_id = _get_secret_or_env("B2_KEY_ID")
    b2_app_key = _get_secret_or_env("B2_APP_KEY")
    b2_bucket = _get_secret_or_env("B2_BUCKET_NAME", "pensieve-db")
    b2_prefix = _get_secret_or_env("B2_PREFIX", "chroma_db/")

    if not b2_key_id or not b2_app_key:
        st.error("B2 credentials not configured (need B2_KEY_ID and B2_APP_KEY).")
        st.stop()

    ph = st.empty()
    ph.info("🧠 Loading Pensieve database from B2 (first load may take a minute)…")

    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo

        # Normalize prefix
        b2_prefix = (b2_prefix or "").strip()
        if b2_prefix and not b2_prefix.endswith("/"):
            b2_prefix += "/"

        # Wipe target folder before download (prevents stale/partial DB issues)
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", b2_key_id, b2_app_key)
        bucket = b2_api.get_bucket_by_name(b2_bucket)

        file_count = 0
        for file_version, _ in bucket.ls(folder_to_list=b2_prefix, recursive=True):
            file_name = file_version.file_name
            if not file_name.startswith(b2_prefix):
                continue

            local_rel = file_name[len(b2_prefix) :]
            if not local_rel:
                continue

            local_path = chroma_dir / local_rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            bucket.download_file_by_name(file_name).save_to(str(local_path))
            file_count += 1

        if not any(chroma_dir.iterdir()):
            ph.error(f"Downloaded 0 files; DB folder is empty: {chroma_dir}")
            st.stop()

        marker_file.touch()
        ph.success(f"✅ Database loaded ({file_count} files)")
        return chroma_dir

    except Exception as e:
        ph.error(f"Failed to download database from B2: {e}")
        st.stop()


# ---------- Embedding / retrieval helpers ----------
def embed_query(client: OpenAI, text: str, model: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def run_query(col, q_emb: list[float], top_k: int) -> dict:
    return col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


# ---------- Theme persistence helpers ----------
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


# ---------- AI query-focused summary helpers ----------
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
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.output_text or "").strip()
    except Exception:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (chat.choices[0].message.content or "").strip()


# ---------- UI helpers ----------
def inject_css(mode: str) -> None:
    if mode == "dark":
        bg = "#0f111a"
        panel = "#141827"
        border = "rgba(255,255,255,0.10)"
        text = "rgba(255,255,255,0.92)"
        muted = "rgba(255,255,255,0.70)"
        link = "#7aa2ff"
        input_bg = "rgba(255,255,255,0.06)"
        pill_bg = "rgba(255,255,255,0.06)"
        hover_bg = "rgba(255,255,255,0.04)"
    else:
        bg = "#fbf7ef"
        panel = "#ffffff"
        border = "rgba(0,0,0,0.08)"
        text = "rgba(0,0,0,0.88)"
        muted = "rgba(0,0,0,0.62)"
        link = "#1f5fbf"
        input_bg = "rgba(0,0,0,0.03)"
        pill_bg = "rgba(0,0,0,0.03)"
        hover_bg = "rgba(0,0,0,0.03)"

    st.markdown(
        f"""
        <style>
        .block-container {{
            max-width: 1400px;
            padding-left: 3rem;
            padding-right: 3rem;
        }}
        .stApp {{
            background: {bg};
            color: {text};
        }}
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                        Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif;
        }}
        a, a:visited {{
            color: {link} !important;
        }}
        .muted {{
            color: {muted};
        }}
        .stExpander {{
            border-radius: 12px;
            border: 1px solid {border};
            background: {panel};
        }}
        div[data-baseweb="input"] > div {{
            background: {input_bg};
        }}
        div[data-baseweb="select"] > div {{
            background: {input_bg};
        }}
        .stNumberInput, .stSelectbox {{
            margin-top: -2px;
        }}
        .pensieve-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid {border};
            background: {pill_bg};
        }}
        div[data-testid="stExpander"] > details > summary {{
            background-color: transparent !important;
        }}
        div[data-testid="stExpander"] summary > div {{
            background-color: transparent !important;
        }}
        div[data-testid="stExpander"] summary:focus,
        div[data-testid="stExpander"] summary:active {{
            background-color: transparent !important;
        }}
        div[data-testid="stExpander"] summary:hover {{
            background-color: {hover_bg} !important;
        }}
        .pensieve-meta {{ margin: 0 0 0.35rem 0; }}
        .pensieve-kv {{ margin: 0.15rem 0; line-height: 1.25; }}
        .pensieve-kv .k {{ font-weight: 600; opacity: 0.9; }}
        .pensieve-kv .v {{ opacity: 0.95; }}
        .pensieve-divider {{
            height: 1px;
            background: {border};
            margin: 0.55rem 0;
        }}
        div.stButton > button {{
            background: {input_bg} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
        }}
        div.stButton > button:hover {{
            background: {pill_bg} !important;
        }}
        div.stButton > button:active {{
            transform: translateY(0.5px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def fetch_note_summaries_by_doc_ids(col_sum, doc_ids: list[str]) -> dict[str, str]:
    if col_sum is None or not doc_ids:
        return {}
    seen = set()
    doc_ids = [d for d in doc_ids if d and not (d in seen or seen.add(d))]
    sids = [f"{did}::summary" for did in doc_ids]
    got = col_sum.get(ids=sids, include=["documents", "metadatas"])
    out: dict[str, str] = {}
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    for sid, doc, meta in zip(ids, docs, metas):
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
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    for sid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        sf = meta.get("source_file") or (sid[:-len("::summary")] if sid.endswith("::summary") else sid)
        out[sf] = (doc or "").strip()
    return out


def render_summary_text(summary_text: str) -> None:
    s = (summary_text or "").strip()
    if not s:
        st.markdown("_No summary found for this item yet._")
        return
    st.markdown(s)


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

        left = folder if folder else "Notes"
        loc_line = f"{left} / {filename}" if folder else filename
        if page_str:
            loc_line += f" • {page_str}"

        header_main = _display_header_from_meta(meta, label)
        display = header_main if header_main else loc_line

        exp_title = f"[{i}] {display}"
        if page_str:
            exp_title += f" • {page_str}"

        if debug:
            if chunk_ix is not None:
                exp_title += f" • chunk={chunk_ix} • d={dist:.3f}"
            else:
                exp_title += f" • d={dist:.3f}"

        key = (meta.get("doc_id") or "").strip() if label == "NOTES" else (meta.get("source_file") or "").strip()
        summary_text = (summaries or {}).get(key, "")

        def _kv(k: str, value: str) -> str:
            v = (value or "").strip()
            if not v:
                return ""
            return f"<div class='pensieve-kv'><span class='k'>{k}:</span> <span class='v'>{v}</span></div>"

        if source_type == "note":
            body_html = (
                "<div class='pensieve-meta'>"
                + _kv("Theme", theme)
                + _kv("Title", paper_title)
                + _kv("Authors", paper_authors)
                + _kv("Year", paper_year)
                + "</div>"
            )
        else:
            body_html = (
                "<div class='pensieve-meta'>"
                + _kv("Title", pdf_title)
                + _kv("Authors", pdf_authors)
                + _kv("Year", (meta.get("year") or "").strip())
                + "</div>"
            )

        if body_html == "<div class='pensieve-meta'></div>":
            body_html = "<div class='pensieve-meta muted'>No metadata available.</div>"

        with st.expander(exp_title, expanded=False):
            st.markdown(body_html, unsafe_allow_html=True)
            st.markdown("<div class='pensieve-divider'></div>", unsafe_allow_html=True)

            if summary_text:
                st.markdown("**Summary**")
                render_summary_text(summary_text)
            else:
                st.markdown("_No summary found for this item yet._")
                if debug and key:
                    st.code(f"Expected summary id: {key}::summary")

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

                show_key = f"show_qsum::{btn_key}"
                if st.session_state.get(show_key, False):
                    MAX_CHARS = 18000
                    doc_text_capped = doc_text[:MAX_CHARS]
                    with st.spinner("Summarizing…"):
                        snippet = cached_query_focused_summary(
                            ck,
                            model=ai_model,
                            query=query,
                            label=label,
                            doc_label=doc_label,
                            doc_text=doc_text_capped,
                        )
                    st.markdown(snippet if snippet else "_No output returned._")

            show_meta = (label == "PAPERS") or bool(heading_path) or debug
            if show_meta:
                st.markdown("<div class='pensieve-divider'></div>", unsafe_allow_html=True)
                with st.expander("Metadata", expanded=False):
                    def _kv2(k: str, value: str) -> str:
                        v = (value or "").strip()
                        if not v:
                            return ""
                        return f"<div class='pensieve-kv'><span class='k'>{k}:</span> <span class='v'>{v}</span></div>"

                    meta_html2 = (
                        "<div class='pensieve-meta'>"
                        + _kv2("Heading path", heading_path)
                        + _kv2("Source file", src)
                        + _kv2("Source type", source_type)
                        + _kv2("Title", meta.get("title") or meta.get("paper_title") or "")
                        + _kv2("Authors", meta.get("authors") or meta.get("paper_authors") or "")
                        + _kv2("Year", meta.get("year") or meta.get("paper_year") or "")
                        + _kv2("Doc ID", (meta.get("doc_id") or "").strip())
                        + _kv2("Chunk index", str(chunk_ix) if chunk_ix is not None else "")
                        + _kv2("Page", str(page) if page else "")
                        + "</div>"
                    )
                    if meta_html2 == "<div class='pensieve-meta'></div>":
                        meta_html2 = "<div class='pensieve-meta muted'>No extra metadata available.</div>"
                    st.markdown(meta_html2, unsafe_allow_html=True)
                    if debug:
                        st.markdown("**Raw meta**")
                        st.json(meta)


# ---------- App ----------
def main():
    st.set_page_config(page_title="Pensieve", layout="wide")

    _init_theme_from_query_params()
    inject_css(st.session_state.get("theme_mode", "light"))

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or _get_secret_or_env("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()

    cfg = load_config()
    repo = _repo_root()

    # DB path is decided here
    chroma_dir = ensure_chroma_db_available()

    top_k_default = int((cfg.get("retrieval") or {}).get("top_k", 8))
    embed_model = cfg.get("embedding_model", "text-embedding-3-small")
    ai_model = ((cfg.get("app", {}) or {}).get("query_snippet_model") or "gpt-4.1-mini").strip()

    oa = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=str(chroma_dir))

    notes = chroma.get_or_create_collection(name="notes_docs")
    try:
        papers = chroma.get_or_create_collection(name="papers_docs")
    except Exception:
        papers = None

    note_summaries = chroma.get_or_create_collection(name="note_summaries")
    paper_summaries = chroma.get_or_create_collection(name="paper_summaries")

    github_url = (cfg.get("app", {}) or {}).get("github_url", "").strip() or "https://github.com/<your-org>/<your-repo>"
    website_url = (cfg.get("app", {}) or {}).get("website_url", "").strip() or ""

    # ---------- HERO ----------
    left, right = st.columns([2.1, 1.4], gap="large")

    with left:
        h1, toggles = st.columns([4, 1.25], vertical_alignment="center")
        with h1:
            st.markdown(
                """
                <h1 style="margin: 0 0 0.15em 0; font-size: 2.35rem;">🧠 Pensieve</h1>
                """,
                unsafe_allow_html=True,
            )
        with toggles:
            c1, c2 = st.columns([1, 1], vertical_alignment="center")
            with c1:
                if st.button("🌗", key="btn_theme", help="Toggle light/dark", use_container_width=True):
                    st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
                    _persist_theme_to_query_params(st.session_state.theme_mode)
                    st.rerun()
            with c2:
                debug = st.toggle("🐛", key="toggle_debug", value=False, help="Debug mode")
        st.markdown(
            """
            <div style="font-size: 1.18em; line-height: 1.5; margin-bottom: 0.55em;">
              A memory basin for <strong>papers</strong> and <strong>notes</strong>.
            </div>
            <div class="muted" style="font-size: 1.02em; line-height: 1.55; margin-bottom: 0.55em;">
              A lightweight semantic search + summarization tool for research workflows.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="muted" style="font-size: 0.98em; line-height: 1.4;">
              🔗 Code: <a href="{github_url}" target="_blank" rel="noopener noreferrer">{github_url}</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if website_url:
            st.markdown(
                f"""
                <div class="muted" style="font-size: 0.98em; line-height: 1.4; margin-top: 0.15em;">
                  🌐 Website:
                  <a href="{website_url}" target="_blank" rel="noopener noreferrer">{website_url}</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if debug:
            st.caption(f"mode={get_deployment_mode()}  chroma_dir={chroma_dir}")

    with right:
        img_path = repo / "assets" / "pensieve.jpg"
        if img_path.exists():
            img = Image.open(img_path)
            st.image(img, use_container_width=True)

    # ---------- CONTROLS ----------
    c_search, c_notes, c_papers, c_k, c_ai = st.columns([2.6, 0.9, 0.9, 1.2, 1.4], vertical_alignment="bottom")

    with c_search:
        q = st.text_input(
            "🔍 Search",
            placeholder='e.g., motivated reasoning, political incivility, epistemic vigilance',
            label_visibility="collapsed",
            key="q",
        )
    with c_notes:
        use_notes = st.checkbox("📝 Notes", value=True, key="use_notes")
    with c_papers:
        use_papers = st.checkbox("📄 Papers", value=True, key="use_papers")
    with c_k:
        top_k_ui = st.number_input(
            "Number of results",
            min_value=1,
            max_value=25,
            value=top_k_default,
            step=1,
            key="top_k_ui",
        )
    with c_ai:
        enable_ai_snippets = st.checkbox(
            "✨ AI snippets",
            value=False,
            key="enable_ai_snippets",
            help="Generate query-focused insights per result",
        )

    # ---------- QUERY ----------
    if q:
        q_emb = embed_query(oa, q, model=embed_model)

        if use_notes:
            res_notes = run_query(notes, q_emb, int(top_k_ui))
            note_doc_ids = [(m or {}).get("doc_id") for m in res_notes["metadatas"][0] if (m or {}).get("doc_id")]
            note_sum_map = fetch_note_summaries_by_doc_ids(note_summaries, note_doc_ids)
            render_results(
                "NOTES",
                res_notes,
                query=q,
                ai_model=ai_model,
                summaries=note_sum_map,
                debug=debug,
                enable_ai_snippets=enable_ai_snippets,
            )

        if use_papers:
            if papers is None:
                st.warning("No 'papers_docs' collection found yet.")
            else:
                res_papers = run_query(papers, q_emb, int(top_k_ui))
                paper_source_files = [
                    (m or {}).get("source_file") for m in res_papers["metadatas"][0] if (m or {}).get("source_file")
                ]
                paper_sum_map = fetch_paper_summaries_by_source_files(paper_summaries, paper_source_files)
                render_results(
                    "PAPERS",
                    res_papers,
                    query=q,
                    ai_model=ai_model,
                    summaries=paper_sum_map,
                    debug=debug,
                    enable_ai_snippets=enable_ai_snippets,
                )

    st.markdown("---")
    st.caption("Made with Streamlit ✨")


if __name__ == "__main__":
    main()