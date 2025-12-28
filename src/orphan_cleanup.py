"""
orphan_cleanup.py

Reusable module for detecting and cleaning orphaned index entries.
Can be imported by update_and_deploy.py or run standalone.

Usage as module:
    from orphan_cleanup import cleanup_orphans
    cleanup_orphans(dry_run=False, verbose=True)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple, Optional
from dataclasses import dataclass

import yaml
import chromadb


@dataclass
class CleanupReport:
    """Summary of orphan detection and cleanup."""
    orphaned_paper_registry_keys: List[str]
    orphaned_note_registry_keys: List[str]
    orphaned_in_chroma: Dict[str, List[str]]  # collection_name -> [source_files]
    orphaned_paper_caches: List[Path]
    orphaned_note_caches: List[Path]
    
    @property
    def total_orphans(self) -> int:
        return (
            len(self.orphaned_paper_registry_keys) +
            len(self.orphaned_note_registry_keys) +
            sum(len(v) for v in self.orphaned_in_chroma.values()) +
            len(self.orphaned_paper_caches) +
            len(self.orphaned_note_caches)
        )
    
    @property
    def has_orphans(self) -> bool:
        return self.total_orphans > 0


def load_config() -> Dict[str, Any]:
    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def get_existing_source_files(papers_dir: Path, notes_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Scan source directories and return sets of relative file paths."""
    existing_papers: Set[str] = set()
    existing_notes: Set[str] = set()

    if papers_dir.exists():
        for pdf_path in papers_dir.rglob("*.pdf"):
            rel = pdf_path.relative_to(papers_dir)
            existing_papers.add(str(rel).replace("\\", "/"))

    if notes_dir.exists():
        for docx_path in notes_dir.rglob("*.docx"):
            rel = docx_path.relative_to(notes_dir)
            existing_notes.add(str(rel).replace("\\", "/"))

    return existing_papers, existing_notes


def find_orphaned_registry_entries(
    registry: Dict[str, Any],
    existing_papers: Set[str],
    existing_notes: Set[str],
) -> Tuple[List[str], List[str]]:
    """Find registry keys referencing deleted files."""
    orphaned_papers = []
    orphaned_notes = []

    for key in registry.keys():
        if key.startswith("papers::"):
            rel_path = key[len("papers::"):]
            if rel_path not in existing_papers:
                orphaned_papers.append(key)
        elif key.startswith("notes::"):
            rel_path = key[len("notes::"):]
            if rel_path not in existing_notes:
                orphaned_notes.append(key)

    return orphaned_papers, orphaned_notes


def find_orphaned_chroma_entries(
    chroma_client: chromadb.ClientAPI,
    collection_name: str,
    existing_files: Set[str],
) -> List[str]:
    """Find ChromaDB entries whose source_file no longer exists."""
    try:
        col = chroma_client.get_collection(name=collection_name)
    except Exception:
        return []

    all_data = col.get(include=["metadatas"])
    metadatas = all_data.get("metadatas") or []

    indexed_files: Set[str] = set()
    for meta in metadatas:
        if meta:
            sf = meta.get("source_file")
            if sf:
                indexed_files.add(sf)

    return list(indexed_files - existing_files)


def find_orphaned_summary_caches(cache_dir: Path, existing_files: Set[str]) -> List[Path]:
    """Find JSON cache files that don't correspond to existing source files."""
    orphaned: List[Path] = []

    if not cache_dir.exists():
        return orphaned

    for cache_file in cache_dir.glob("*.json"):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            summary = data.get("summary", {})
            source_file = summary.get("source_file", "")
            
            if source_file and source_file not in existing_files:
                orphaned.append(cache_file)
        except Exception:
            pass

    return orphaned


def detect_orphans(
    papers_dir: Path,
    notes_dir: Path,
    chroma_dir: Path,
    processed_dir: Path,
    registry_path: Path,
) -> CleanupReport:
    """
    Detect all orphaned entries across registry, ChromaDB, and caches.
    Returns a CleanupReport with all findings.
    """
    existing_papers, existing_notes = get_existing_source_files(papers_dir, notes_dir)
    registry = load_registry(registry_path)

    # Registry orphans
    orphaned_paper_keys, orphaned_note_keys = find_orphaned_registry_entries(
        registry, existing_papers, existing_notes
    )

    # ChromaDB orphans
    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    orphaned_in_chroma: Dict[str, List[str]] = {}

    for col_name in ["papers", "papers_docs", "paper_summaries"]:
        orphaned_in_chroma[col_name] = find_orphaned_chroma_entries(
            chroma, col_name, existing_papers
        )

    for col_name in ["notes", "notes_docs", "note_summaries"]:
        orphaned_in_chroma[col_name] = find_orphaned_chroma_entries(
            chroma, col_name, existing_notes
        )

    # Cache orphans
    paper_cache_dir = processed_dir / "paper_summaries"
    note_cache_dir = processed_dir / "note_summaries"
    orphaned_paper_caches = find_orphaned_summary_caches(paper_cache_dir, existing_papers)
    orphaned_note_caches = find_orphaned_summary_caches(note_cache_dir, existing_notes)

    return CleanupReport(
        orphaned_paper_registry_keys=orphaned_paper_keys,
        orphaned_note_registry_keys=orphaned_note_keys,
        orphaned_in_chroma=orphaned_in_chroma,
        orphaned_paper_caches=orphaned_paper_caches,
        orphaned_note_caches=orphaned_note_caches,
    )


def execute_cleanup(
    report: CleanupReport,
    chroma_dir: Path,
    registry_path: Path,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Execute cleanup based on a CleanupReport.
    Returns dict of {action: count} for items removed.
    """
    results: Dict[str, int] = {}
    
    chroma = chromadb.PersistentClient(path=str(chroma_dir))

    # Remove from ChromaDB collections
    for col_name, orphaned_files in report.orphaned_in_chroma.items():
        if orphaned_files:
            try:
                col = chroma.get_collection(name=col_name)
                for sf in orphaned_files:
                    if verbose:
                        print(f"  Removing from {col_name}: {sf}")
                    col.delete(where={"source_file": sf})
                results[f"chroma_{col_name}"] = len(orphaned_files)
            except Exception as e:
                print(f"  ! Error cleaning {col_name}: {e}")

    # Remove cache files
    for cf in report.orphaned_paper_caches:
        if verbose:
            print(f"  Removing cache: {cf.name}")
        try:
            cf.unlink()
        except Exception:
            pass
    results["paper_caches"] = len(report.orphaned_paper_caches)

    for cf in report.orphaned_note_caches:
        if verbose:
            print(f"  Removing cache: {cf.name}")
        try:
            cf.unlink()
        except Exception:
            pass
    results["note_caches"] = len(report.orphaned_note_caches)

    # Remove registry entries
    all_orphaned_keys = (
        report.orphaned_paper_registry_keys + 
        report.orphaned_note_registry_keys
    )
    if all_orphaned_keys:
        registry = load_registry(registry_path)
        for key in all_orphaned_keys:
            if verbose:
                print(f"  Removing registry: {key}")
            registry.pop(key, None)
        save_registry(registry_path, registry)
        results["registry"] = len(all_orphaned_keys)

    return results


def cleanup_orphans(
    dry_run: bool = True,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> CleanupReport:
    """
    Main entry point for orphan cleanup.
    
    Args:
        dry_run: If True, only detect orphans without removing them
        verbose: If True, print detailed information
        config: Optional config dict (loads from config.yaml if not provided)
    
    Returns:
        CleanupReport with all findings
    """
    from dotenv import load_dotenv
    load_dotenv()

    cfg = config or load_config()
    target = (os.getenv("PENSIEVE_TARGET", "") or "").strip().lower()
    repo = Path(__file__).resolve().parent.parent

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (repo / pp)

    papers_dir = _abs(cfg["paths"]["papers_dir"])
    notes_dir = _abs(cfg["paths"]["notes_dir"])

    chosen_chroma = (
        (os.getenv("PENSIEVE_CHROMA_DIR") or "").strip()
        or (cfg.get("paths", {}) or {}).get("chroma_dir")
        or (
            (cfg.get("paths", {}) or {}).get("chroma_dir_server")
            if target == "server"
            else (cfg.get("paths", {}) or {}).get("chroma_dir_local")
        )
    )
    chroma_dir = _abs(chosen_chroma)

    chosen_processed = (
        (os.getenv("PENSIEVE_PROCESSED_DIR") or "").strip()
        or cfg["paths"]["processed_dir"]
    )
    processed_dir = _abs(chosen_processed)

    chosen_registry = (
        (os.getenv("PENSIEVE_REGISTRY_PATH") or "").strip()
        or (
            cfg["paths"]["registry_server"]
            if target == "server"
            else cfg["paths"]["registry_local"]
        )
    )
    registry_path = _abs(chosen_registry)

    # Detect orphans
    report = detect_orphans(
        papers_dir=papers_dir,
        notes_dir=notes_dir,
        chroma_dir=chroma_dir,
        processed_dir=processed_dir,
        registry_path=registry_path,
    )

    if verbose:
        print(f"\nOrphan Detection Report:")
        print(f"  Registry - papers: {len(report.orphaned_paper_registry_keys)}")
        print(f"  Registry - notes:  {len(report.orphaned_note_registry_keys)}")
        for col_name, orphans in report.orphaned_in_chroma.items():
            print(f"  ChromaDB {col_name}: {len(orphans)}")
        print(f"  Paper caches: {len(report.orphaned_paper_caches)}")
        print(f"  Note caches:  {len(report.orphaned_note_caches)}")
        print(f"  TOTAL: {report.total_orphans}")

    # Execute cleanup if not dry run
    if not dry_run and report.has_orphans:
        if verbose:
            print("\nExecuting cleanup...")
        results = execute_cleanup(
            report=report,
            chroma_dir=chroma_dir,
            registry_path=registry_path,
            verbose=verbose,
        )
        if verbose:
            print(f"\nCleanup results: {results}")

    return report


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and clean orphaned index entries")
    parser.add_argument("--execute", action="store_true", help="Actually remove orphans")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    report = cleanup_orphans(dry_run=not args.execute, verbose=args.verbose)
    
    if report.has_orphans:
        if args.execute:
            print(f"\n✓ Cleaned up {report.total_orphans} orphaned entries")
        else:
            print(f"\n⚠ Found {report.total_orphans} orphans. Run with --execute to remove.")
    else:
        print("\n✓ No orphans found. Index is clean!")