from __future__ import annotations

from pathlib import Path
import yaml
import chromadb


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    chroma_dir = Path(cfg["paths"]["chroma_dir"])

    chroma = chromadb.PersistentClient(path=str(chroma_dir))

    # choose one:
    col = chroma.get_collection(name="papers_docs")  # or "papers", "notes_docs", "notes"

    # Pull a handful of items (Chroma doesn't have "list all" nicely; query with dummy text is easiest)
    # If this fails because embeddings missing, use col.get below instead.
    got = col.get(limit=5, include=["metadatas", "documents"])

    for i in range(len(got["ids"])):
        print("\n---")
        print("ID:", got["ids"][i])
        print("META:", got["metadatas"][i])

if __name__ == "__main__":
    main()