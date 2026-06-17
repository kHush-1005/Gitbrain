"""
scripts/export_chunks.py  [NEW — Week 2]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    The main Week 2 pipeline script.

    Fetches a GitHub repository, chunks all code files, and saves the
    result as a JSONL file in data/chunks/.

    Week 3 will read this JSONL file to generate embeddings — so producing
    a clean, complete JSONL here is the primary goal of Week 2.

USAGE:
    python scripts/export_chunks.py https://github.com/owner/repo
    python scripts/export_chunks.py https://github.com/tiangolo/fastapi
    python scripts/export_chunks.py https://github.com/psf/requests --max-files 200

OUTPUT:
    data/chunks/{owner}__{repo}_chunks.jsonl  ← one chunk dict per line
    data/chunks/{owner}__{repo}_summary.json  ← ingestion stats

ENVIRONMENT:
    GITHUB_TOKEN in .env  (optional but recommended for 5,000 req/hr limit)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import logging
import argparse
from collections import Counter

# ── Make sure the project root is on sys.path ──────────────────────────────────
# This allows running the script from any directory as:
#   python scripts/export_chunks.py ...
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from utils.repo_parser import parse_github_url
from utils.file_utils  import (
    save_jsonl, save_json, get_chunk_file_path,
    sanitize_repo_name, ensure_dir
)
from core.github_ingester import (
    fetch_repository_files, RepoNotFoundError,
    GitHubRateLimitError, GitHubAPIError
)
from core.code_chunker import chunk_all_files

# ─── Logging (set to WARNING so info noise is hidden during normal runs) ──────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GitBrain Week 2 — Fetch, chunk, and export a GitHub repository."
    )
    parser.add_argument(
        "repo_url",
        help="GitHub repository URL, e.g. https://github.com/tiangolo/fastapi"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=500,
        help="Maximum number of files to fetch (default: 500)."
    )
    parser.add_argument(
        "--output-dir",
        default="data/chunks",
        help="Directory to save chunk files (default: data/chunks)."
    )
    return parser.parse_args()


def print_header(owner: str, repo: str) -> None:
    print()
    print("=" * 70)
    print("  GitBrain — Week 2 Export Chunks")
    print("=" * 70)
    print(f"  Repository : {owner}/{repo}")
    print(f"  Output dir : data/chunks/")
    print("=" * 70)


def print_chunk_stats(chunks: list[dict]) -> None:
    """Print a breakdown of chunk types and languages."""
    if not chunks:
        return

    type_counts = Counter(c["chunk_type"] for c in chunks)
    lang_counts = Counter(c["language"]   for c in chunks)

    char_counts     = [c["char_count"]     for c in chunks]
    token_estimates = [c["token_estimate"] for c in chunks]

    print("\n  ── Chunk Type Breakdown ─────────────────────────────────────")
    for ctype, count in type_counts.most_common():
        bar = "█" * min(count // 2, 36)
        print(f"    {ctype:<20} {count:>5}  {bar}")

    print("\n  ── Language Breakdown ───────────────────────────────────────")
    for lang, count in lang_counts.most_common(10):
        bar = "█" * min(count // 2, 36)
        print(f"    {lang:<20} {count:>5}  {bar}")

    avg_chars   = sum(char_counts)   / len(char_counts)
    avg_tokens  = sum(token_estimates) / len(token_estimates)
    max_chars   = max(char_counts)
    min_chars   = min(char_counts)

    print(f"\n  ── Size Statistics ──────────────────────────────────────────")
    print(f"    Total chunks       : {len(chunks):>6}")
    print(f"    Avg chunk size     : {avg_chars:>6.0f} chars  (~{avg_tokens:.0f} tokens)")
    print(f"    Largest chunk      : {max_chars:>6} chars")
    print(f"    Smallest chunk     : {min_chars:>6} chars")

    oversized = sum(1 for c in chunks if c["token_estimate"] > 512)
    if oversized:
        print(f"\n  ⚠  {oversized} chunk(s) have token_estimate > 512.")
        print("     Run validate_chunks.py to inspect them.")
    else:
        print(f"    All chunks within 512-token limit  ✓")


def run_export(repo_url: str, max_files: int, output_dir: str) -> None:
    """
    Main pipeline:
        parse URL → fetch files → chunk → save JSONL → save summary
    """
    start_time = time.time()

    # ── Step 1: Parse URL ──────────────────────────────────────────────────────
    print("\n  [1/5] Parsing repository URL...")
    try:
        owner, repo = parse_github_url(repo_url)
    except ValueError as e:
        print(f"\n  ✗ ERROR: {e}")
        sys.exit(1)

    repo_name = sanitize_repo_name(owner, repo)
    print(f"        Owner     : {owner}")
    print(f"        Repo      : {repo}")
    print(f"        Safe name : {repo_name}")

    # ── Step 2: Read token ─────────────────────────────────────────────────────
    print("\n  [2/5] Checking GitHub authentication...")
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        print(f"        ✓ Token found (first 8 chars: {token[:8]}...)")
        print(f"          Rate limit: 5,000 requests/hour")
    else:
        print("        ⚠  No GITHUB_TOKEN in .env — rate limit: 60 req/hour")
        print("           Add your token to .env for faster ingestion.")

    # ── Step 3: Fetch files ────────────────────────────────────────────────────
    print("\n  [3/5] Fetching files from GitHub...")
    try:
        files, ingest_summary = fetch_repository_files(
            owner=owner, repo=repo, token=token, max_files=max_files
        )
    except RepoNotFoundError as e:
        print(f"\n  ✗ Repository Error:\n    {e}")
        sys.exit(1)
    except GitHubRateLimitError as e:
        print(f"\n  ✗ Rate Limit Error:\n    {e}")
        sys.exit(1)
    except GitHubAPIError as e:
        print(f"\n  ✗ GitHub API Error:\n    {e}")
        sys.exit(1)

    if not files:
        print("\n  ✗ No files fetched. Cannot generate chunks.")
        sys.exit(1)

    print(f"\n        ✓ {len(files)} files fetched with content")

    # ── Step 4: Chunk files ────────────────────────────────────────────────────
    print("\n  [4/5] Chunking files...")
    chunks = chunk_all_files(files, repo_name=repo_name)

    if not chunks:
        print("\n  ✗ No chunks generated. Files may be empty.")
        sys.exit(1)

    print(f"\n        ✓ {len(chunks)} chunks generated")

    # ── Step 5: Save to disk ───────────────────────────────────────────────────
    print(f"\n  [5/5] Saving chunks to disk...")
    ensure_dir(output_dir)

    # Save JSONL (one chunk per line)
    jsonl_path = get_chunk_file_path(owner, repo, output_dir)
    written    = save_jsonl(chunks, jsonl_path)

    # Save summary JSON alongside the JSONL
    summary = {
        **ingest_summary,
        "repo_name":       repo_name,
        "total_chunks":    len(chunks),
        "output_file":     jsonl_path,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    summary_path = jsonl_path.replace("_chunks.jsonl", "_summary.json")
    save_json(summary, summary_path)

    print(f"\n        ✓ Saved {written} chunks → {jsonl_path}")
    print(f"        ✓ Saved summary       → {summary_path}")

    # ── Print statistics ───────────────────────────────────────────────────────
    print_chunk_stats(chunks)

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"  ✓ EXPORT COMPLETE")
    print(f"  {len(chunks)} chunks from {len(files)} files in {elapsed:.1f} seconds")
    print()
    print("  NEXT: Validate the chunks:")
    print(f"    python scripts/validate_chunks.py {jsonl_path}")
    print("=" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    print_header(*parse_github_url(args.repo_url) if "--" not in args.repo_url else ("?", "?"))
    run_export(args.repo_url, args.max_files, args.output_dir)
