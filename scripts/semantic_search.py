"""
scripts/semantic_search.py  [NEW — Week 3]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Query the vector index with natural language and see the most relevant
    code chunks returned — all from the terminal, before Week 4 adds the
    FastAPI + LLM layer.

    This lets you validate that the vector index is working correctly:
    asking "how does authentication work?" should return auth-related code,
    not database schema code.

USAGE (Windows):
    python scripts\semantic_search.py <repo_name> "<your question>"

    python scripts\semantic_search.py psf__requests "how are headers set?"
    python scripts\semantic_search.py tiangolo__fastapi "how does routing work?"
    python scripts\semantic_search.py tiangolo__fastapi "explain the dependency injection"

OPTIONS:
    --top-k N           Number of results to show. Default: 5.
    --language LANG     Filter results to one language (e.g. python).
    --min-score FLOAT   Only show results with score ≥ this. Default: 0.0.
    --show-code         Print the full chunk text (not just preview). Default: False.

PREREQUISITES:
    Run build_vector_index.py first to create the index.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from core.embedder     import embed_text
from core.vector_store import similarity_search, collection_count, list_collections

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# ─── Preview configuration ────────────────────────────────────────────────────
PREVIEW_CHARS = 200   # characters of chunk text to show in preview mode


def parse_args():
    parser = argparse.ArgumentParser(
        description="GitBrain Week 3 — Semantic search over a vector-indexed repository."
    )
    parser.add_argument(
        "repo_name",
        help="Collection name (e.g. psf__requests, tiangolo__fastapi)"
    )
    parser.add_argument(
        "query",
        help="Natural language search query (use quotes for multi-word queries)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--language", default=None,
        help="Filter results by language (e.g. --language python)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Minimum similarity score threshold (0.0–1.0, default: 0.0)"
    )
    parser.add_argument(
        "--show-code", action="store_true",
        help="Print full chunk text instead of a short preview"
    )
    return parser.parse_args()


def format_score_bar(score: float, width: int = 20) -> str:
    """Render a visual bar for the similarity score."""
    filled = int(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    return bar


def print_result(result: dict, index: int, show_code: bool) -> None:
    """Pretty-print a single search result."""
    score     = result["score"]
    file_path = result["file_path"]
    name      = result["chunk_name"]
    ctype     = result["chunk_type"]
    lang      = result["language"]
    start     = result["start_line"]
    end       = result["end_line"]
    tokens    = result["token_estimate"]
    text      = result["text"]

    bar = format_score_bar(score)

    print(f"\n  ┌─ Result #{index + 1} " + "─" * 48)
    print(f"  │  Score      : {score:.4f}  [{bar}]")
    print(f"  │  File       : {file_path}")
    print(f"  │  Name       : {name}")
    print(f"  │  Type       : {ctype}  |  Language: {lang}")
    print(f"  │  Lines      : {start} – {end}  (~{tokens} tokens)")

    if show_code:
        print(f"  │  Code:")
        for line in text.splitlines():
            print(f"  │    {line}")
    else:
        preview = text.replace("\n", " ↵ ")[:PREVIEW_CHARS]
        if len(text) > PREVIEW_CHARS:
            preview += "..."
        print(f"  │  Preview    : {preview!r}")

    print(f"  └" + "─" * 60)


def run_search(
    repo_name:  str,
    query:      str,
    top_k:      int,
    language:   str | None,
    min_score:  float,
    show_code:  bool,
) -> bool:
    """
    Execute semantic search and print results.
    Returns True if results found, False otherwise.
    """
    print()
    print("=" * 70)
    print("  GitBrain — Week 3: Semantic Search")
    print("=" * 70)
    print(f"  Repository : {repo_name}")
    print(f"  Query      : {query!r}")
    print(f"  Top-K      : {top_k}")
    if language:
        print(f"  Language   : {language} (filter active)")

    # ── Check collection exists ────────────────────────────────────────────────
    count = collection_count(repo_name)
    if count == 0:
        print(f"\n  ✗ ERROR: Collection '{repo_name}' is empty or doesn't exist.")
        print("  Run build_vector_index.py first:")
        print(f"    python scripts\\build_vector_index.py data\\chunks\\{repo_name}_chunks.jsonl")
        available = list_collections()
        if available:
            print(f"\n  Available collections: {', '.join(available)}")
        return False

    print(f"  Index size : {count} chunks")
    print()

    # ── Embed query ────────────────────────────────────────────────────────────
    print("  [1/2] Embedding query...")
    start = time.time()
    query_vector = embed_text(query)
    embed_time   = (time.time() - start) * 1000
    print(f"        ✓ Query embedded in {embed_time:.0f}ms")
    print(f"          Vector dim: {len(query_vector)}")

    # ── Search ChromaDB ────────────────────────────────────────────────────────
    print(f"\n  [2/2] Searching collection '{repo_name}'...")
    where = {"language": language} if language else None

    start   = time.time()
    results = similarity_search(
        query_vector = query_vector,
        repo_name    = repo_name,
        top_k        = top_k,
        where        = where,
    )
    search_time = (time.time() - start) * 1000

    # Apply minimum score filter
    if min_score > 0.0:
        before = len(results)
        results = [r for r in results if r["score"] >= min_score]
        if len(results) < before:
            print(f"        (filtered {before - len(results)} results below min-score {min_score})")

    print(f"        ✓ Search completed in {search_time:.0f}ms")

    # ── Print results ──────────────────────────────────────────────────────────
    if not results:
        print(f"\n  No results found.")
        if min_score > 0.0:
            print(f"  Try lowering --min-score (currently {min_score})")
        return False

    print(f"\n  Found {len(results)} result(s) for: {query!r}")
    for i, result in enumerate(results):
        print_result(result, i, show_code)

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        best  = results[0]
        worst = results[-1]
        print(f"\n  ── Score Range ──────────────────────────────────────────")
        print(f"    Best match  : {best['score']:.4f}  — {best['file_path']} → {best['chunk_name']}")
        print(f"    Worst match : {worst['score']:.4f}  — {worst['file_path']} → {worst['chunk_name']}")
        if best["score"] < 0.3:
            print("\n  ⚠  Low scores — the query may not match well with this codebase.")
            print("     Try a more specific or technical query.")
        elif best["score"] > 0.7:
            print("\n  ✓ High confidence results!")

    print("=" * 70)
    print()

    # Hint for Week 4
    print("  Week 4 will pass these chunks to Llama 3.1 (via Groq) to generate")
    print("  a natural-language answer with file citations.")
    print()
    return True


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print()
        print("Usage:")
        print("  python scripts\\semantic_search.py <repo_name> \"<query>\"")
        print()
        print("Examples:")
        print("  python scripts\\semantic_search.py psf__requests \"how are headers set?\"")
        print("  python scripts\\semantic_search.py tiangolo__fastapi \"how does routing work?\"")
        print("  python scripts\\semantic_search.py psf__requests \"authentication\" --language python --top-k 3")
        print()
        available = list_collections()
        if available:
            print(f"Available collections: {', '.join(available)}")
        sys.exit(1)

    args    = parse_args()
    success = run_search(
        repo_name  = args.repo_name,
        query      = args.query,
        top_k      = args.top_k,
        language   = args.language,
        min_score  = args.min_score,
        show_code  = args.show_code,
    )
    sys.exit(0 if success else 1)
