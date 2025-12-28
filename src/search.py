from __future__ import annotations

import os
from pathlib import Path
import yaml

from dotenv import load_dotenv
from openai import OpenAI
import chromadb


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def embed_query(client: OpenAI, text: str, model: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def pretty_print_results(label: str, res: dict, limit: int = 8) -> None:
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print(f"\n=== {label} (top {min(limit, len(docs))}) ===")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        src = meta.get("source_file", "unknown")
        chunk_ix = meta.get("chunk_index", "?")
        page = meta.get("page_num")  # None for notes
        page_str = f" | p.{page}" if page else ""

        paper_title = meta.get("title") or ""
        authors = meta.get("authors") or ""
        meta_line = ""
        if paper_title or authors:
            meta_line = f"{paper_title}" + (f" — {authors}" if authors else "")

        preview = doc[:260].replace("\n", " ")
        print(f"\n[{i}] {src}{page_str} | chunk {chunk_ix} | distance={dist:.4f}")
        if meta_line:
            print(f"    {meta_line}")
        print(preview + ("…" if len(doc) > 260 else ""))

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    cfg = load_config()
    chroma_dir = Path(cfg["paths"]["chroma_dir"])
    top_k = int(cfg["retrieval"].get("top_k", 8))
    embed_model = cfg.get("embedding_model", "text-embedding-3-small")

    oa = OpenAI(api_key=api_key)

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    notes = chroma.get_collection(name="notes")
    papers = None
    try:
        papers = chroma.get_collection(name="papers")
    except Exception:
        pass

    mode = "both"  # notes | papers | both
    print("Search mode: notes | papers | both (type /mode to change). Blank query quits.")
    print(f"Current mode: {mode}")

    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break

        if q.lower().startswith("/mode"):
            parts = q.split()
            if len(parts) == 2 and parts[1] in {"notes", "papers", "both"}:
                mode = parts[1]
                print(f"Mode set to: {mode}")
            else:
                print("Usage: /mode notes|papers|both")
            continue

        q_emb = embed_query(oa, q, model=embed_model)

        if mode in {"notes", "both"}:
            res_notes = notes.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            pretty_print_results("NOTES", res_notes, limit=top_k)

        if mode in {"papers", "both"}:
            if papers is None:
                print("\n=== PAPERS ===\nNo 'papers' collection found yet. Run: python src/index_papers.py")
            else:
                res_papers = papers.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                pretty_print_results("PAPERS", res_papers, limit=top_k)


if __name__ == "__main__":
    main()
