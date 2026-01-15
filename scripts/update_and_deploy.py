"""
Pensieve Update & Deploy Script
================================

One-click script to index, summarize, and optionally deploy your Pensieve database.

====================================================================
LOCAL USE (Default - No cloud deployment)
-----------------------------------------
Most users will just want to index locally and run the app:

    # Full pipeline (index + summarize)
    python scripts/update_and_deploy.py --skip-upload

    # Only process notes
    python scripts/update_and_deploy.py --notes-only --skip-upload

    # Only process papers
    python scripts/update_and_deploy.py --papers-only --skip-upload

    # Preview changes without executing
    python scripts/update_and_deploy.py --dry-run

    # Force full reprocess (ignore cache)
    python scripts/update_and_deploy.py --force --skip-upload

    # Clean up orphaned entries after deleting files
    python scripts/update_and_deploy.py --cleanup --skip-upload

Then run the app:
    streamlit run app/streamlit_app.py

====================================================================
CLOUD DEPLOYMENT (Advanced - Two-repo setup)
--------------------------------------------
For users who want to deploy to Streamlit Cloud with a separate live repo:

    # Full rebuild + publish to live repo + upload to B2
    python scripts/update_and_deploy.py --live-repo "/path/to/pensieve-live" --sync-deletions

    # Publish current DB only (no re-indexing)
    python scripts/update_and_deploy.py --live-repo "/path/to/pensieve-live" --skip-index --skip-summarize --sync-deletions

    # Copy local → live DB only (no B2 upload)
    python scripts/update_and_deploy.py --live-repo "/path/to/pensieve-live" --copy-only

    # Upload only (live repo already updated)
    python scripts/update_and_deploy.py --live-repo "/path/to/pensieve-live" --upload-only --sync-deletions

====================================================================
FLAGS REFERENCE
---------------
Indexing control:
    --skip-index        Skip indexing steps
    --skip-summarize    Skip summarization steps
    --notes-only        Only process notes (skip papers)
    --papers-only       Only process papers (skip notes)
    --force             Force full reprocess (ignore cache)
    --cleanup           Clean orphaned entries before indexing

Deployment control:
    --skip-upload       Skip B2 upload (local only)
    --upload-only       Only upload to B2 (skip index/summarize/copy)
    --copy-only         Only copy local DB to live repo (no upload)
    --live-repo PATH    Path to separate live deployment repo
    --sync-deletions    Delete B2 files that no longer exist locally

Other:
    --dry-run           Show what would happen without making changes

====================================================================
AFTER UPLOADING TO B2
---------------------
Streamlit Cloud → Manage app → Clear cache → Reboot

====================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


# ============ B2 CONFIG ============
B2_BUCKET_NAME = "pensieve-db"  # Override via B2_BUCKET_NAME env var
B2_PREFIX = "chroma_db"         # Override via B2_PREFIX env var
# ==================================


def log(message: str, level: str = "INFO") -> None:
    """Print a timestamped log message with emoji indicator."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARN": "⚠️ ",
        "UPLOAD": "📤",
        "DELETE": "🗑️ ",
        "SKIP": "⏭️ ",
        "START": "🚀",
        "DRY": "👀",
        "CLEAN": "🧹",
        "COPY": "📁",
    }
    icon = icons.get(level, "  ")
    print(f"[{timestamp}] {icon} {message}")


def run_script(script_path: Path, description: str, extra_args: list[str] | None = None) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'─'*60}")
    log(description, "START")
    print(f"{'─'*60}")

    if not script_path.exists():
        log(f"Script not found: {script_path}", "ERROR")
        return False

    start_time = time.time()
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, cwd=script_path.parent.parent)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        log(f"{description} failed with code {result.returncode} ({elapsed:.1f}s)", "ERROR")
        return False

    log(f"{description} complete ({elapsed:.1f}s)", "SUCCESS")
    return True


def format_size(n: int) -> str:
    """Format byte size as human-readable string."""
    size = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def sha1_file(path: Path) -> str:
    """Compute SHA1 hash of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_chroma_db(source_db: Path, target_db: Path, dry_run: bool = False) -> bool:
    """
    Wipe target_db and copy source_db into it.
    Used for copying local DB to live repo's server folder.
    """
    if not source_db.exists():
        log(f"Source ChromaDB not found: {source_db}", "ERROR")
        return False

    source_files = [p for p in source_db.rglob("*") if p.is_file()]
    source_size = sum(p.stat().st_size for p in source_files)
    log(f"Source DB: {len(source_files)} files, {format_size(source_size)}", "INFO")
    log(f"Copy target: {target_db}", "COPY")

    if dry_run:
        log("DRY RUN: would wipe target and copy DB", "DRY")
        return True

    # Wipe and recreate
    if target_db.exists():
        shutil.rmtree(target_db)
    target_db.mkdir(parents=True, exist_ok=True)

    # Copy tree
    shutil.copytree(source_db, target_db, dirs_exist_ok=True)

    # Sanity check
    if not any(target_db.iterdir()):
        log(f"Copy completed but target is empty: {target_db}", "ERROR")
        return False

    log("Copied source DB -> target folder", "SUCCESS")
    return True


def upload_to_b2(chroma_path: Path, sync_deletions: bool = False, dry_run: bool = False) -> bool:
    """
    Upload the given chroma_path folder to Backblaze B2 (incremental).
    Only uploads new or changed files based on SHA1 comparison.
    """
    print(f"\n{'='*60}")
    log("Upload ChromaDB to Backblaze B2 (Incremental)", "START")
    if dry_run:
        log("DRY RUN - no changes will be made", "DRY")
    print(f"{'='*60}")

    if not chroma_path.exists():
        log(f"ChromaDB not found at: {chroma_path}", "ERROR")
        return False

    local_files = {f.relative_to(chroma_path).as_posix(): f for f in chroma_path.rglob("*") if f.is_file()}
    local_size = sum(f.stat().st_size for f in local_files.values())
    log(f"Uploading from: {chroma_path}")
    log(f"Local ChromaDB: {len(local_files)} files, {format_size(local_size)}")

    # Get B2 credentials
    b2_key_id = os.getenv("B2_KEY_ID") or os.getenv("B2_APPLICATION_KEY_ID")
    b2_app_key = os.getenv("B2_APP_KEY") or os.getenv("B2_APPLICATION_KEY")
    b2_bucket = os.getenv("B2_BUCKET_NAME", B2_BUCKET_NAME)
    b2_prefix = os.getenv("B2_PREFIX", B2_PREFIX).rstrip("/")
    
    if not b2_key_id or not b2_app_key:
        log("B2 credentials not found. Set B2_KEY_ID and B2_APP_KEY in .env", "ERROR")
        return False

    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
    except ImportError:
        log("b2sdk not installed. Run: pip install b2sdk", "ERROR")
        return False

    try:
        log("Authenticating with B2...")
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", b2_key_id, b2_app_key)
        log("Authentication successful", "SUCCESS")

        bucket = b2_api.get_bucket_by_name(b2_bucket)
        log(f"Bucket: {b2_bucket}")
        log(f"Prefix: {b2_prefix}/")

        # List existing B2 files
        log("Fetching existing B2 file list...")
        b2_files: dict[str, dict] = {}
        for file_version, _ in bucket.ls(folder_to_list=f"{b2_prefix}/", recursive=True):
            rel_name = file_version.file_name
            if rel_name.startswith(f"{b2_prefix}/"):
                rel_name = rel_name[len(f"{b2_prefix}/"):]
            if rel_name:
                b2_files[rel_name] = {
                    "sha1": file_version.content_sha1,
                    "size": file_version.size,
                    "id": file_version.id_,
                    "full_name": file_version.file_name,
                }
        log(f"B2 has {len(b2_files)} existing files")

        stats = {"uploaded": 0, "skipped": 0, "deleted": 0, "bytes_uploaded": 0}
        start_time = time.time()

        # Upload new/changed files
        log("Comparing files...", "UPLOAD")
        for rel_path, local_file in local_files.items():
            b2_path = f"{b2_prefix}/{rel_path}"
            local_hash = sha1_file(local_file)
            local_sz = local_file.stat().st_size

            if rel_path in b2_files:
                b2_hash = b2_files[rel_path]["sha1"]
                if isinstance(b2_hash, str) and b2_hash.startswith("unverified:"):
                    b2_hash = b2_hash[len("unverified:"):]
                if b2_hash == local_hash:
                    stats["skipped"] += 1
                    continue
                action = "UPDATE"
            else:
                action = "NEW"

            if dry_run:
                log(f"  Would {action}: {rel_path} ({format_size(local_sz)})", "DRY")
            else:
                bucket.upload_local_file(local_file=str(local_file), file_name=b2_path)

            stats["uploaded"] += 1
            stats["bytes_uploaded"] += local_sz

            if stats["uploaded"] % 25 == 0:
                log(f"  Uploaded {stats['uploaded']} files...", "UPLOAD")

        # Delete orphaned files from B2
        if sync_deletions:
            orphaned = set(b2_files.keys()) - set(local_files.keys())
            if orphaned:
                log(f"Found {len(orphaned)} orphaned files in B2", "DELETE")
                for rel_path in sorted(orphaned):
                    info_dict = b2_files[rel_path]
                    if dry_run:
                        log(f"  Would DELETE: {rel_path}", "DRY")
                    else:
                        bucket.delete_file_version(info_dict["id"], info_dict["full_name"])
                    stats["deleted"] += 1
            else:
                log("No orphaned files to delete", "INFO")

        elapsed = time.time() - start_time
        print(f"\n{'─'*60}")
        if dry_run:
            log("DRY RUN SUMMARY (no changes made):", "DRY")
        else:
            log("Upload complete!", "SUCCESS")

        log(f"  Uploaded: {stats['uploaded']} files ({format_size(stats['bytes_uploaded'])})")
        log(f"  Skipped (unchanged): {stats['skipped']} files")
        if sync_deletions:
            log(f"  Deleted (orphaned): {stats['deleted']} files")
        log(f"  Time: {elapsed:.1f}s")
        if stats["uploaded"] > 0 and elapsed > 0:
            speed = stats["bytes_uploaded"] / elapsed
            log(f"  Speed: {format_size(speed)}/s")

        print(f"{'─'*60}")
        return True

    except Exception as e:
        log(f"B2 operation failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def run_cleanup(src_dir: Path, dry_run: bool = False) -> bool:
    """Run orphan cleanup to remove stale index entries."""
    orphan_cleanup_path = src_dir / "orphan_cleanup.py"
    if not orphan_cleanup_path.exists():
        log(f"orphan_cleanup.py not found at: {orphan_cleanup_path}", "ERROR")
        return False

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("orphan_cleanup", orphan_cleanup_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["orphan_cleanup"] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        report = mod.cleanup_orphans(dry_run=dry_run, verbose=True)
        if getattr(report, "has_orphans", False):
            if dry_run:
                log(f"Would remove {report.total_orphans} orphaned entries", "DRY")
            else:
                log(f"Cleaned up {report.total_orphans} orphaned entries", "SUCCESS")
        else:
            log("No orphans found - index is clean", "SUCCESS")
        return True
    except Exception as e:
        log(f"Cleanup failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update and deploy Pensieve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local indexing only (most common)
  python scripts/update_and_deploy.py --skip-upload

  # Index only notes
  python scripts/update_and_deploy.py --notes-only --skip-upload

  # Force full reindex
  python scripts/update_and_deploy.py --force --skip-upload

  # Deploy to cloud (with live repo)
  python scripts/update_and_deploy.py --live-repo /path/to/live-repo --sync-deletions
        """
    )
    
    # Indexing control
    parser.add_argument("--skip-index", action="store_true", 
                        help="Skip indexing steps")
    parser.add_argument("--skip-summarize", action="store_true", 
                        help="Skip summarization steps")
    parser.add_argument("--notes-only", action="store_true", 
                        help="Only process notes (skip papers)")
    parser.add_argument("--papers-only", action="store_true", 
                        help="Only process papers (skip notes)")
    parser.add_argument("--force", action="store_true", 
                        help="Force full reprocess (ignore cache)")
    parser.add_argument("--cleanup", action="store_true", 
                        help="Clean orphaned entries before indexing")
    
    # Deployment control
    parser.add_argument("--skip-upload", action="store_true", 
                        help="Skip B2 upload (local only)")
    parser.add_argument("--upload-only", action="store_true", 
                        help="Only upload to B2 (skip index/summarize/copy)")
    parser.add_argument("--copy-only", action="store_true", 
                        help="Only copy local DB to live repo (no upload)")
    parser.add_argument("--live-repo", type=str, default=None,
                        help="Path to separate live deployment repo (for two-repo setup)")
    parser.add_argument("--sync-deletions", action="store_true", 
                        help="Delete B2 files that no longer exist locally")
    
    # Other
    parser.add_argument("--dry-run", action="store_true", 
                        help="Show what would happen without making changes")
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Determine paths
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    local_db = repo_root / "data" / "_local" / "chroma_db"

    # Determine upload source path
    if args.live_repo:
        live_root = Path(args.live_repo).expanduser().resolve()
        live_db = live_root / "data" / "_server" / "chroma_db"
        upload_source = live_db
    else:
        live_root = None
        live_db = None
        upload_source = local_db

    # Print header
    print(f"\n{'='*60}")
    print("🧠 PENSIEVE UPDATE & DEPLOY")
    print(f"{'='*60}")
    log(f"Repository root: {repo_root}")
    log(f"Local DB path:   {local_db}")
    if live_root:
        log(f"Live repo root:  {live_root}")
        log(f"Live DB path:    {live_db}")
    if args.dry_run:
        log("DRY RUN MODE - no changes will be made", "DRY")

    overall_start = time.time()
    success = True

    # Handle shortcut flags
    if args.upload_only:
        args.skip_index = True
        args.skip_summarize = True

    # Validate conflicting options
    if args.notes_only and args.papers_only:
        log("Cannot use both --notes-only and --papers-only", "ERROR")
        return 1

    # Step 0: Cleanup (optional)
    if args.cleanup:
        print(f"\n{'─'*60}")
        log("Cleaning orphaned index entries", "CLEAN")
        print(f"{'─'*60}")
        if not run_cleanup(src_dir, dry_run=args.dry_run):
            success = False

    # Step 1: Index notes
    if not args.skip_index and not args.papers_only:
        extra_args = ["--force"] if args.force else None
        if not run_script(src_dir / "index_notes.py", "Indexing notes", extra_args):
            success = False
    elif args.papers_only:
        log("Skipping notes (--papers-only)", "SKIP")
    else:
        log("Skipping indexing (--skip-index)", "SKIP")

    # Step 2: Index papers
    if not args.skip_index and not args.notes_only:
        extra_args = ["--force"] if args.force else None
        if not run_script(src_dir / "index_papers.py", "Indexing papers", extra_args):
            success = False
    elif args.notes_only:
        log("Skipping papers (--notes-only)", "SKIP")

    # Step 3: Summarize notes
    if not args.skip_summarize and not args.papers_only:
        if not run_script(src_dir / "summarize_notes.py", "Summarizing notes"):
            success = False
    elif args.papers_only:
        log("Skipping note summaries (--papers-only)", "SKIP")
    else:
        log("Skipping summarization (--skip-summarize)", "SKIP")

    # Step 4: Summarize papers
    if not args.skip_summarize and not args.notes_only:
        if not run_script(src_dir / "summarize_papers.py", "Summarizing papers"):
            success = False
    elif args.notes_only:
        log("Skipping paper summaries (--notes-only)", "SKIP")

    # Step 5: Copy to live repo (if using two-repo setup)
    if live_db and not args.upload_only:
        if not copy_chroma_db(local_db, live_db, dry_run=args.dry_run):
            success = False
    elif args.upload_only and live_db:
        log("Skipping copy step (--upload-only)", "SKIP")

    # Early exit for copy-only mode
    if args.copy_only:
        overall_elapsed = time.time() - overall_start
        log(f"Copy-only complete ({overall_elapsed:.1f}s)", "SUCCESS" if success else "ERROR")
        return 0 if success else 1

    # Step 6: Upload to B2
    if not args.skip_upload:
        if not upload_to_b2(upload_source, sync_deletions=args.sync_deletions, dry_run=args.dry_run):
            success = False
    else:
        log("Skipping B2 upload (--skip-upload)", "SKIP")

    # Final summary
    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    log(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")

    if success:
        log("All steps completed successfully!", "SUCCESS")
        if not args.dry_run and not args.skip_upload:
            print("\n📝 Next: Streamlit Cloud → Manage app → Clear cache → Reboot.")
        elif not args.dry_run:
            print("\n📝 Next: Run 'streamlit run app/streamlit_app.py' to start the app.")
    else:
        log("Some steps failed. Check the output above.", "WARN")

    print(f"{'='*60}\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())