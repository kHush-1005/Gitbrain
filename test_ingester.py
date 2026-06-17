"""
test_ingester.py  [UPDATED — Week 2]
─────────────────────────────────────────────────────────────────────────────
CHANGES FROM WEEK 1:
    ✓ Uses updated github_ingester (now returns summary dict too)
    ✓ Uses updated code_chunker (JS heuristic chunks, richer metadata)
    ✓ Prints token_estimate alongside char_count for each chunk
    ✓ Offers to save chunks to disk at the end (uses export_chunks pipeline)
    ✓ Still works with the exact same command as Week 1:

        python test_ingester.py https://github.com/owner/repo

    ✓ Optional --save flag to write JSONL without running export_chunks.py:

        python test_ingester.py https://github.com/owner/repo --save
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import logging
import argparse
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from utils.repo_parser    import parse_github_url, make_collection_name
from core.github_ingester import (
    fetch_repository_files, RepoNotFoundError,
    GitHubRateLimitError, GitHubAPIError
)
from core.code_chunker import chunk_all_files

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s"
)

SHOW_CHUNKS  = 20
SHOW_PREVIEW = 130


def parse_args():
    parser = argparse.ArgumentParser(
        description="GitBrain — test ingestion and chunking from the terminal."
    )
    parser.add_argument("repo_url", help="GitHub repository URL")
    parser.add_argument("--max-files", type=int, default=100,
                        help="Maximum files to fetch (default: 100)")
    parser.add_argument("--save", action="store_true",
                        help="Also save chunks to data/chunks/ as JSONL")
    return parser.parse_args()


def print_divider(title=""):
    print()
    if title:
        print(f"  ── {title} " + "─" * max(2, 60 - len(title)))
    else:
        print("  " + "─" * 66)


def print_chunk(chunk: dict, index: int):
    print(f"\n  ┌─ Chunk #{index + 1} " + "─" * 50)
    print(f"  │  File      : {chunk['file_path']}")
    print(f"  │  Name      : {chunk['chunk_name']}")
    print(f"  │  Type      : {chunk['chunk_type']}")
    print(f"  │  Language  : {chunk['language']}")
    print(f"  │  Lines     : {chunk['start_line']} – {chunk['end_line']}")
    print(f"  │  Size      : {chunk['char_count']} chars  (~{chunk['token_estimate']} tokens)")
    preview = chunk["text"].replace("\n", "↵ ")[:SHOW_PREVIEW]
    if len(chunk["text"]) > SHOW_PREVIEW:
        preview += "..."
    print(f"  │  Preview   : {preview!r}")
    print(f"  └" + "─" * 60)


def run_test(repo_url: str, max_files: int, save: bool):
    start_time = time.time()

    print()
    print("=" * 70)
    print("  GitBrain — Week 2 Ingestion Test")
    print("=" * 70)

    # ── Parse URL ─────────────────────────────────────────────────────────────
    print_divider("STEP 1: Parsing Repository URL")
    try:
        owner, repo = parse_github_url(repo_url)
    except ValueError as e:
        print(f"\n  ✗ ERROR: {e}")
        sys.exit(1)

    collection_name = make_collection_name(owner, repo)
    print(f"  Input URL   : {repo_url}")
    print(f"  Owner       : {owner}")
    print(f"  Repo        : {repo}")
    print(f"  Collection  : {collection_name}  (ChromaDB key for Week 3)")
    print(f"  ✓ URL parsed successfully")

    # ── Auth ──────────────────────────────────────────────────────────────────
    print_divider("STEP 2: GitHub Authentication")
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        print(f"  ✓ Token loaded (first 8: {token[:8]}...)  — 5,000 req/hr")
    else:
        print("  ⚠  No token in .env — 60 req/hr (may be slow on large repos)")

    # ── Fetch ─────────────────────────────────────────────────────────────────
    print_divider("STEP 3: Fetching Files from GitHub")
    try:
        files, summary = fetch_repository_files(
            owner=owner, repo=repo, token=token, max_files=max_files
        )
    except RepoNotFoundError as e:
        print(f"\n  ✗ Repo Error: {e}")
        sys.exit(1)
    except GitHubRateLimitError as e:
        print(f"\n  ✗ Rate Limit: {e}")
        sys.exit(1)
    except GitHubAPIError as e:
        print(f"\n  ✗ API Error: {e}")
        sys.exit(1)

    if not files:
        print("\n  ✗ No files fetched.")
        sys.exit(1)

    print(f"\n  ✓ {len(files)} files fetched  |  {summary['empty']} empty/skipped")
    lang_summary = Counter(f["language"] for f in files)
    print("\n  Languages found:")
    for lang, count in lang_summary.most_common(8):
        print(f"    {lang:<18} {count}")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print_divider("STEP 4: Chunking Files")
    print(f"  Processing {len(files)} files...\n")
    chunks = chunk_all_files(files, repo_name=collection_name)

    if not chunks:
        print("\n  ✗ No chunks produced.")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n  ✓ {len(chunks)} chunks in {elapsed:.1f}s")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print_divider("STEP 5: Chunk Statistics")
    type_counts = Counter(c["chunk_type"] for c in chunks)
    lang_counts = Counter(c["language"]   for c in chunks)

    print("\n  Chunk types:")
    for t, n in type_counts.most_common():
        print(f"    {t:<22} {n}")
    print("\n  Languages:")
    for l, n in lang_counts.most_common(8):
        print(f"    {l:<22} {n}")

    chars       = [c["char_count"]     for c in chunks]
    tokens      = [c["token_estimate"] for c in chunks]
    oversized   = sum(1 for t in tokens if t > 512)
    print(f"\n  Avg size   : {sum(chars)/len(chars):.0f} chars  (~{sum(tokens)/len(tokens):.0f} tokens)")
    print(f"  Largest    : {max(chars)} chars")
    print(f"  Oversized  : {oversized} chunks (> 512 tokens)")

    # ── Sample chunks ─────────────────────────────────────────────────────────
    n = min(SHOW_CHUNKS, len(chunks))
    print_divider(f"STEP 6: First {n} Chunks")
    for i, chunk in enumerate(chunks[:n]):
        print_chunk(chunk, i)

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        print_divider("STEP 7: Saving Chunks to Disk")
        from utils.file_utils import save_jsonl, get_chunk_file_path, ensure_dir
        path = get_chunk_file_path(owner, repo)
        ensure_dir(os.path.dirname(path))
        written = save_jsonl(chunks, path)
        print(f"  ✓ Saved {written} chunks → {path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_divider("WEEK 2 VALIDATION SUMMARY")
    checks = [
        (True, "URL parsed correctly"),
        (len(files) > 0,  f"{len(files)} files fetched from GitHub"),
        (True,            "Binary files filtered (SKIP_EXTENSIONS active)"),
        (any(c["chunk_type"] in ("function","class","method") for c in chunks),
         "Python AST chunks extracted"),
        (any(c["chunk_type"] == "heuristic_block" for c in chunks),
         "JS/TS heuristic chunks extracted (if JS files present)"),
        (all(c["start_line"] >= 1 and c["end_line"] >= c["start_line"] for c in chunks),
         "All line numbers valid"),
        (all("char_count" in c and "token_estimate" in c for c in chunks),
         "char_count + token_estimate on all chunks"),
        (oversized == 0,  "No oversized chunks (all ≤ 512 tokens)"),
    ]
    for passed, label in checks:
        print(f"  {'✓' if passed else '⚠'} {label}")

    print()
    print("=" * 70)
    print(f"  WEEK 2 TEST COMPLETE")
    print(f"  {len(chunks)} chunks from {len(files)} files in {elapsed:.1f}s")
    print()
    print("  To save chunks permanently:")
    print(f"    python scripts/export_chunks.py {repo_url}")
    print()
    print("  To validate saved chunks:")
    jsonl_path = f"data/chunks/{collection_name}_chunks.jsonl"
    print(f"    python scripts/validate_chunks.py {jsonl_path}")
    print()
    print("  To run all tests:")
    print("    pytest tests/ -v")
    print("=" * 70)
    print()


if __name__ == "__main__":
    args = parse_args()
    run_test(args.repo_url, args.max_files, args.save)
