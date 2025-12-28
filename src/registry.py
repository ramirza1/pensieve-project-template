from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def file_fingerprint(p: Path) -> Dict[str, Any]:
    st = p.stat()
    return {"mtime": st.st_mtime, "size": st.st_size}


def is_unchanged(registry: Dict[str, Any], key: str, fp: Dict[str, Any]) -> bool:
    old = registry.get(key)
    return bool(old) and old.get("mtime") == fp["mtime"] and old.get("size") == fp["size"]