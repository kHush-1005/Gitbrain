"""
scripts/build_vector_index.py  [NEW — Week 3]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    The main Week 3 script.

    Reads the JSONL chunk file produced by Week 2's export_chunks.py,
    generates an embedding vector for each chunk, and stores everything
    in ChromaDB for semantic search.

    After this script completes, you can run semantic_search.py to
    query the indexed repository using natural language.

USAGE (Windows):
    python scripts\build_vector_index.py data\chunks\psf__requests_chunks.jsonl
    python scripts\build_vector_index.py data\chunks\tiangolo__fastapi_chunks.jsonl

USAGE (Mac/Linux):
    python scripts/build_vector_index.py data/chunks/psf__requests_chunks.jsonl

OPTIONS:
    --batch-size N    Chunks to embed at once. Default: 32.
                      Reduce to 8-16 if you run out of memory.
    --fresh           Delete existing collection before indexing.
                      Use this to re-index from scratch.
    --repo-name NAME  Override the repository name used as the collection name.
                      Default: derived from the JSONL filename.

OUTPUT:
    data/chroma_db/   ChromaDB collection files saved here.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path
from collections import Counter

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from utils.file_utils  import load_jsonl
from core.embedder     import embed_batch, get_embedding_dim, get_embedding_model
from core.vector_store import (
    upsert_chunks, collection_count, delete_collection,
    sanitize_collection_name
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="GitBrain Week 3 — Build vector index from JSONL chunks."
    )
    parser.add_argument(
        "jsonl_path",
        help="Path to the JSONL chunk file from Week 2.\n"
             "Example: data\\chunks\\psf__requests_chunks.jsonl"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Chunks per embedding batch (default: 32). Reduce if memory issues."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete existing collection before indexing (full re-index)."
    )
    parser.add_argument(
        "--repo-name", default=None,
        help="Override collection name. Default: derived from JSONL filename."
    )
    return parser.parse_args()


def derive_repo_name_from_path(jsonl_path: str) -> str:
    """
    Extract repo name from JSONL filename.

    Example:
        "data/chunks/psf__requests_chunks.jsonl" → "psf__requests"
        "data\\chunks\\tiangolo__fastapi_chunks.jsonl" → "tiangolo__fastapi"
    """
    stem = Path(jsonl_path).stem          # e.g. "psf__requests_chunks"
    name = stem.replace("_chunks", "")    # e.g. "psf__requests"
    return name


def print_header():
    print()
    print("=" * 70)
    print("  GitBrain — Week 3: Build Vector Index")
    print("=" * 70)


def print_embedding_stats(chunks: list, elapsed: float, batch_size: int):
    """Print statistics after embedding."""
    total_chars  = sum(c.get("char_count", len(c["text"])) for c in chunks)
    total_tokens = sum(c.get("token_estimate", 0) for c in chunks)
    type_counts  = Counter(c.get("chunk_type","?") for c in chunks)

    print(f"\n  Embedding stats:")
    print(f"    Chunks embedded    : {len(chunks)}")
    print(f"    Total characters   : {total_chars:,}")
    print(f"    ~Total tokens      : {total_tokens:,}")
    print(f"    Batch size used    : {batch_size}")
    print(f"    Time to embed      : {elapsed:.1f}s")
    speed = len(chunks) / elapsed if elapsed > 0 else 0
    print(f"    Speed              : {speed:.0f} chunks/sec")
    print(f"\n  Chunk type breakdown:")
    for ctype, count in type_counts.most_common():
        bar = "█" * min(count // 3, 30)
        print(f"    {ctype:<22} {count:>5}  {bar}")


def run_build_index(
    jsonl_path: str,
    batch_size: int,
    fresh:      bool,
    repo_name:  str | None,
) -> bool:
    """
    Main indexing pipeline: load → embed → store.
    Returns True on success, False on failure.
    """
    print_header()
    start_total = time.time()

    # ── Resolve paths using pathlib (Windows + Linux safe) ────────────────────
    jsonl_path = str(Path(jsonl_path))   # normalize separators

    # ── Step 1: Derive repo name ───────────────────────────────────────────────
    print("\n  [1/5] Setting up...")
    if repo_name is None:
        repo_name = derive_repo_name_from_path(jsonl_path)

    safe_name = sanitize_collection_name(repo_name)
    print(f"        JSONL file   : {jsonl_path}")
    print(f"        Repo name    : {repo_name}")
    print(f"        Collection   : {safe_name}")

    # ── Step 2: Optionally delete existing collection ─────────────────────────
    if fresh:
        print(f"\n  [--fresh] Deleting existing collection '{safe_name}'...")
        deleted = delete_collection(repo_name)
        if deleted:
            print(f"           ✓ Deleted")
        else:
            print(f"           (no existing collection found)")

    # ── Step 3: Load JSONL chunks ──────────────────────────────────────────────
    print(f"\n  [2/5] Loading chunks from JSONL...")
    try:
        chunks = load_jsonl(jsonl_path)
    except FileNotFoundError as e:
        print(f"\n  ✗ ERROR: {e}")
        print("  Run this first:")
        print(f"    python scripts\\export_chunks.py https://github.com/OWNER/REPO")
        return False

    if not chunks:
        print("\n  ✗ ERROR: JSONL file is empty. Re-run export_chunks.py.")
        return False

    print(f"        ✓ Loaded {len(chunks)} chunks")

    # Check if already indexed
    existing = collection_count(repo_name)
    if existing > 0 and not fresh:
        print(f"\n  ⚠  Collection already contains {existing} chunks.")
        print("      Use --fresh to re-index from scratch.")
        print("      Proceeding with upsert (duplicates will be updated)...")

    # ── Step 4: Generate embeddings ────────────────────────────────────────────
    print(f"\n  [3/5] Generating embeddings...")
    print(f"        Embedding dimension : {get_embedding_dim()}")
    print(f"        Batch size          : {batch_size}")
    print(f"        Model               : {os.getenv('EMBEDDING_MODEL','all-MiniLM-L6-v2')}")

    # Load model upfront so the "loading..." message appears before the progress
    get_embedding_model()

    texts       = [c["text"] for c in chunks]
    start_embed = time.time()

    print(f"\n        Embedding {len(texts)} chunks...")
    vectors = embed_batch(texts, batch_size=batch_size, show_progress=True)

    embed_elapsed = time.time() - start_embed
    print(f"\n        ✓ Embeddings generated in {embed_elapsed:.1f}s")

    # Attach embeddings to chunks
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    # Validate a sample
    sample_dim = len(chunks[0]["embedding"])
    if sample_dim != get_embedding_dim():
        print(f"\n  ✗ ERROR: Expected {get_embedding_dim()}-dim embeddings, got {sample_dim}")
        return False

    print_embedding_stats(chunks, embed_elapsed, batch_size)

    # ── Step 5: Store in ChromaDB ──────────────────────────────────────────────
    print(f"\n  [4/5] Storing in ChromaDB...")
    start_store = time.time()
    stored = upsert_chunks(chunks, repo_name)
    store_elapsed = time.time() - start_store

    print(f"        ✓ {stored} chunks stored in {store_elapsed:.1f}s")

    # ── Step 6: Verify ────────────────────────────────────────────────────────
    print(f"\n  [5/5] Verifying...")
    final_count = collection_count(repo_name)
    print(f"        Collection '{safe_name}' contains {final_count} chunks")

    if final_count == 0:
        print("\n  ✗ ERROR: Collection is empty after indexing. Something went wrong.")
        return False

    total_elapsed = time.time() - start_total
    print()
    print("=" * 70)
    print("  ✓ VECTOR INDEX BUILD COMPLETE")
    print(f"  {final_count} chunks indexed in {total_elapsed:.1f} seconds")
    print()
    print("  NEXT: Run semantic search:")
    print(f"    python scripts\\semantic_search.py {safe_name} \"how does login work?\"")
    print("=" * 70)
    print()
    return True


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args    = parse_args()
    success = run_build_index(
        jsonl_path = args.jsonl_path,
        batch_size = args.batch_size,
        fresh      = args.fresh,
        repo_name  = args.repo_name,
    )
    sys.exit(0 if success else 1)
