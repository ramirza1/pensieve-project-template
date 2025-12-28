"""
One-click script to:
0. (Optional) Clean orphaned index entries
1. Re-index notes and papers
2. Re-summarize notes and papers  
3. Upload ChromaDB to Backblaze B2 (incremental - only changed files)
4. Optionally delete orphaned files from B2

Usage:
    python scripts/update_and_deploy.py                    # Full pipeline
    python scripts/update_and_deploy.py --cleanup          # Clean orphans first, then full pipeline
    python scripts/update_and_deploy.py --cleanup-only     # Only clean orphans
    python scripts/update_and_deploy.py --skip-index       # Skip indexing
    python scripts/update_and_deploy.py --skip-summarize   # Skip summarization
    python scripts/update_and_deploy.py --skip-upload      # Skip B2 upload
    python scripts/update_and_deploy.py --upload-only      # Just upload to B2
    python scripts/update_and_deploy.py --sync-deletions   # Also delete orphaned B2 files
    python scripts/update_and_deploy.py --dry-run          # Show what would be uploaded/deleted
"""

import subprocess
import sys
import os
import time
import hashlib
from pathlib import Path
from datetime import datetime

# ============ CONFIGURATION ============
B2_BUCKET_NAME = "pensieve-db"
CHROMA_DIR = "data/_local/chroma_db"
B2_PREFIX = "chroma_db"
# =======================================


def log(message: str, level: str = "INFO") -> None:
    """Print a timestamped log message."""
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
    }
    icon = icons.get(level, "  ")
    print(f"[{timestamp}] {icon} {message}")


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return True if successful."""
    print(f"\n{'─'*60}")
    log(f"{description}", "START")
    print(f"{'─'*60}")
    
    if not script_path.exists():
        log(f"Script not found: {script_path}", "ERROR")
        return False
    
    start_time = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent.parent,
    )
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        log(f"{description} failed with code {result.returncode} ({elapsed:.1f}s)", "ERROR")
        return False
    
    log(f"{description} complete ({elapsed:.1f}s)", "SUCCESS")
    return True


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def get_local_file_hash(file_path: Path) -> str:
    """Calculate SHA1 hash of local file (B2 uses SHA1)."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def upload_to_b2(repo_root: Path, sync_deletions: bool = False, dry_run: bool = False) -> bool:
    """
    Upload ChromaDB folder to Backblaze B2 (incremental).
    
    Args:
        repo_root: Path to the repository root
        sync_deletions: If True, delete B2 files that don't exist locally
        dry_run: If True, show what would happen without making changes
    """
    print(f"\n{'='*60}")
    log("Upload ChromaDB to Backblaze B2 (Incremental)", "START")
    if dry_run:
        log("DRY RUN - no changes will be made", "DRY")
    print(f"{'='*60}")
    
    chroma_path = repo_root / CHROMA_DIR
    
    if not chroma_path.exists():
        log(f"ChromaDB not found at: {chroma_path}", "ERROR")
        return False
    
    # Calculate local DB stats
    local_files = {f.relative_to(chroma_path).as_posix(): f for f in chroma_path.rglob("*") if f.is_file()}
    local_size = sum(f.stat().st_size for f in local_files.values())
    log(f"Local ChromaDB: {len(local_files)} files, {format_size(local_size)}")
    
    # Get B2 credentials
    b2_key_id = os.getenv("B2_KEY_ID")
    b2_app_key = os.getenv("B2_APP_KEY")
    
    if not b2_key_id or not b2_app_key:
        log("B2 credentials not found. Set B2_KEY_ID and B2_APP_KEY in .env", "ERROR")
        return False
    
    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
    except ImportError:
        log("b2sdk not installed. Run: pip install b2sdk", "ERROR")
        return False
    
    try:
        # Authenticate
        log("Authenticating with B2...")
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", b2_key_id, b2_app_key)
        log("Authentication successful", "SUCCESS")
        
        # Get bucket
        bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        log(f"Bucket: {B2_BUCKET_NAME}")
        
        # Get existing B2 files with their hashes
        log("Fetching existing B2 file list...")
        b2_files = {}
        for file_version, _ in bucket.ls(folder_to_list=f"{B2_PREFIX}/", recursive=True):
            # Remove prefix to get relative path
            rel_name = file_version.file_name
            if rel_name.startswith(f"{B2_PREFIX}/"):
                rel_name = rel_name[len(f"{B2_PREFIX}/"):]
            if rel_name:
                b2_files[rel_name] = {
                    "sha1": file_version.content_sha1,
                    "size": file_version.size,
                    "id": file_version.id_,
                    "full_name": file_version.file_name,
                }
        log(f"B2 has {len(b2_files)} existing files")
        
        # ===== UPLOAD NEW/CHANGED FILES =====
        start_time = time.time()
        stats = {"uploaded": 0, "skipped": 0, "deleted": 0, "bytes_uploaded": 0}
        
        log("Comparing files...", "UPLOAD")
        
        for rel_path, local_file in local_files.items():
            b2_path = f"{B2_PREFIX}/{rel_path}"
            local_hash = get_local_file_hash(local_file)
            local_size = local_file.stat().st_size
            
            # Check if file exists in B2 with same hash
            if rel_path in b2_files:
                b2_hash = b2_files[rel_path]["sha1"]
                # B2 sometimes prefixes hash with "unverified:" for large files
                if b2_hash.startswith("unverified:"):
                    b2_hash = b2_hash[len("unverified:"):]
                
                if b2_hash == local_hash:
                    stats["skipped"] += 1
                    continue
                else:
                    action = "UPDATE"
            else:
                action = "NEW"
            
            # Upload file
            if dry_run:
                log(f"  Would {action}: {rel_path} ({format_size(local_size)})", "DRY")
            else:
                bucket.upload_local_file(
                    local_file=str(local_file),
                    file_name=b2_path,
                )
            
            stats["uploaded"] += 1
            stats["bytes_uploaded"] += local_size
            
            if stats["uploaded"] % 10 == 0:
                log(f"  Uploaded {stats['uploaded']} files...", "UPLOAD")
        
        # ===== DELETE ORPHANED FILES =====
        if sync_deletions:
            orphaned = set(b2_files.keys()) - set(local_files.keys())
            
            if orphaned:
                log(f"Found {len(orphaned)} orphaned files in B2", "DELETE")
                
                for rel_path in orphaned:
                    b2_info = b2_files[rel_path]
                    
                    if dry_run:
                        log(f"  Would DELETE: {rel_path}", "DRY")
                    else:
                        bucket.delete_file_version(b2_info["id"], b2_info["full_name"])
                    
                    stats["deleted"] += 1
            else:
                log("No orphaned files to delete", "INFO")
        
        # ===== SUMMARY =====
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Update and deploy Pensieve")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing")
    parser.add_argument("--skip-summarize", action="store_true", help="Skip summarization")
    parser.add_argument("--skip-upload", action="store_true", help="Skip B2 upload")
    parser.add_argument("--upload-only", action="store_true", help="Only do B2 upload")
    parser.add_argument("--sync-deletions", action="store_true", 
                        help="Delete B2 files that no longer exist locally")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded/deleted without making changes")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove orphaned index entries before indexing")
    parser.add_argument("--cleanup-only", action="store_true",
                        help="Only run cleanup, skip all other steps")
    args = parser.parse_args()
    
    from dotenv import load_dotenv
    load_dotenv()
    
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    
    print(f"\n{'='*60}")
    print("🧠 PENSIEVE UPDATE & DEPLOY")
    print(f"{'='*60}")
    log(f"Repo root: {repo_root}")
    log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        log("DRY RUN MODE - no changes will be made", "DRY")
    
    overall_start = time.time()
    success = True
    
    # Handle --upload-only shortcut
    if args.upload_only:
        args.skip_index = True
        args.skip_summarize = True
    
    # Step 0: Cleanup orphans (if requested)
    if args.cleanup or args.cleanup_only:
        print(f"\n{'─'*60}")
        log("Cleaning orphaned index entries", "CLEAN")
        print(f"{'─'*60}")
        
        # Check if orphan_cleanup.py exists
        orphan_cleanup_path = src_dir / "orphan_cleanup.py"
        if not orphan_cleanup_path.exists():
            log(f"orphan_cleanup.py not found at: {orphan_cleanup_path}", "ERROR")
            log("Make sure orphan_cleanup.py is in your src/ folder", "INFO")
            success = False
        else:
            try:
                # Dynamic import - must register in sys.modules for dataclasses to work
                import importlib.util
                spec = importlib.util.spec_from_file_location("orphan_cleanup", orphan_cleanup_path)
                orphan_module = importlib.util.module_from_spec(spec)
                sys.modules["orphan_cleanup"] = orphan_module  # Required for @dataclass
                spec.loader.exec_module(orphan_module)
                
                report = orphan_module.cleanup_orphans(dry_run=args.dry_run, verbose=True)
                
                if report.has_orphans:
                    if args.dry_run:
                        log(f"Would remove {report.total_orphans} orphaned entries", "DRY")
                    else:
                        log(f"Cleaned up {report.total_orphans} orphaned entries", "SUCCESS")
                else:
                    log("No orphans found - index is clean", "SUCCESS")
                    
            except Exception as e:
                log(f"Cleanup failed: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                success = False
        
        if args.cleanup_only:
            overall_elapsed = time.time() - overall_start
            print(f"\n{'='*60}")
            log(f"Cleanup-only complete ({overall_elapsed:.1f}s)", "SUCCESS" if success else "ERROR")
            print(f"{'='*60}\n")
            return 0 if success else 1
    
    # Step 1: Index
    if not args.skip_index:
        if not run_script(src_dir / "index_notes.py", "Indexing notes"):
            success = False
        if not run_script(src_dir / "index_papers.py", "Indexing papers"):
            success = False
    else:
        log("Skipping indexing", "SKIP")
    
    # Step 2: Summarize
    if not args.skip_summarize:
        if not run_script(src_dir / "summarize_notes.py", "Summarizing notes"):
            success = False
        if not run_script(src_dir / "summarize_papers.py", "Summarizing papers"):
            success = False
    else:
        log("Skipping summarization", "SKIP")
    
    # Step 3: Upload to B2
    if not args.skip_upload:
        if not upload_to_b2(repo_root, sync_deletions=args.sync_deletions, dry_run=args.dry_run):
            success = False
    else:
        log("Skipping B2 upload", "SKIP")
    
    # Summary
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    log(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    
    if success:
        log("All steps completed successfully!", "SUCCESS")
        if not args.dry_run:
            print("\n📝 Your Streamlit app will auto-load the fresh DB on next visit.")
            print(f"   URL: https://pensieve-live.streamlit.app")
    else:
        log("Some steps failed. Check the output above.", "WARN")
    
    print(f"{'='*60}\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())