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


def embed_query(client: OpenAI, text: str, model: str = "text-embedding-3-small") -> list[float]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


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
    col = chroma.get_collection(name="notes")

    print("Type a query (or blank to quit):")
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break

        q_emb = embed_query(oa, q, model=embed_model)

        res = col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        print(f"\nTop {len(docs)} results:")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            src = meta.get("source_file", "unknown")
            chunk_ix = meta.get("chunk_index", "?")
            preview = doc[:260].replace("\n", " ")
            print(f"\n[{i}] {src} | chunk {chunk_ix} | distance={dist:.4f}")
            print(preview + ("…" if len(doc) > 260 else ""))


if __name__ == "__main__":
    main()