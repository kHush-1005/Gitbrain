"""
core/vector_store.py  [NEW — Week 3]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Store and retrieve code chunk embeddings using ChromaDB.

WHAT IS CHROMADB?
    ChromaDB is a local vector database — it stores your chunk text,
    the embedding vector, and metadata (file path, line numbers, etc.)
    all in one place on disk. When you run a semantic search query, it
    uses the cosine distance between vectors to find the most similar chunks.

    Think of it as a search engine where "search terms" are mathematical
    vectors instead of keywords.

    Files are saved in: data/chroma_db/
    One folder per collection. No cloud service needed. Free.

KEY CONCEPTS:
    Collection  — like a database table; one collection per repository
    Document    — the chunk text (what the LLM will read)
    Embedding   — the 384-float vector (what ChromaDB searches on)
    Metadata    — file_path, start_line, end_line, chunk_type, etc.
    ID          — unique identifier per chunk (file_path::chunk_name::start_line)

FUNCTIONS:
    get_client()                   — lazy singleton ChromaDB client
    sanitize_collection_name(name) — make name ChromaDB-safe
    get_or_create_collection(name) — open/create a collection
    upsert_chunks(chunks, name)    — store embedded chunks
    similarity_search(query_vec, name, top_k) — semantic search
    collection_count(name)         — count stored chunks
    delete_collection(name)        — delete a collection (for re-indexing)
    list_collections()             — list all collections in the DB
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Default ChromaDB storage path ────────────────────────────────────────────
# Uses pathlib for Windows/Linux compatibility (no hardcoded slashes)
DEFAULT_CHROMA_DIR = str(Path("data") / "chroma_db")

# ─── ChromaDB collection name rules ───────────────────────────────────────────
# ChromaDB collection names must be 3-63 chars, alphanumeric + underscores only
MIN_COLLECTION_NAME_LEN = 3
MAX_COLLECTION_NAME_LEN = 63

# ─── Lazy singleton ChromaDB client ───────────────────────────────────────────
_chroma_client      = None
_chroma_persist_dir = None


def get_client(persist_dir: str = None):
    """
    Get (or create) the ChromaDB persistent client singleton.

    The client is created once and reused. Data is saved to disk at
    persist_dir after every operation. On restart, all previously stored
    data is immediately available without re-indexing.

    Args:
        persist_dir: Directory path for ChromaDB storage.
                     Defaults to data/chroma_db/ (or CHROMA_PERSIST_DIR env var).

    Returns:
        chromadb.PersistentClient instance.

    Raises:
        ImportError: If chromadb is not installed.
        RuntimeError: If the client cannot be initialized.
    """
    global _chroma_client, _chroma_persist_dir

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)

    # Return existing client if persist_dir hasn't changed
    if _chroma_client is not None and _chroma_persist_dir == persist_dir:
        return _chroma_client

    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "chromadb is not installed.\n"
            "Fix: pip install chromadb\n"
            "     (or: pip install -r requirements.txt)"
        )

    # Ensure the storage directory exists (works on Windows + Linux)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    try:
        _chroma_client      = chromadb.PersistentClient(path=str(persist_dir))
        _chroma_persist_dir = persist_dir
        logger.info(f"ChromaDB client initialized at: {persist_dir}")
        return _chroma_client
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize ChromaDB at '{persist_dir}': {e}\n"
            "Check that the directory is writable and not locked by another process."
        ) from e


def sanitize_collection_name(name: str) -> str:
    """
    Make a string safe for use as a ChromaDB collection name.

    ChromaDB rules:
        - 3 to 63 characters
        - Only alphanumeric characters (a-z, A-Z, 0-9) and underscores
        - Cannot start or end with an underscore or period
        - Cannot contain consecutive periods

    This function:
        1. Lowercases the name
        2. Replaces /, -, . with underscores
        3. Removes any other invalid characters
        4. Truncates to 63 characters
        5. Pads with underscores if shorter than 3 characters

    Args:
        name: Raw name (e.g. "tiangolo/fastapi", "my-org__my-repo")

    Returns:
        A ChromaDB-safe collection name string.

    Example:
        sanitize_collection_name("tiangolo/fastapi")   → "tiangolo__fastapi"
        sanitize_collection_name("My-Org/my.repo.v2")  → "my_org__my_repo_v2"
    """
    # Lowercase everything
    safe = name.lower()

    # Replace common separators with underscores
    safe = safe.replace("/", "__").replace("-", "_").replace(".", "_")

    # Remove any character that is not alphanumeric or underscore
    safe = re.sub(r"[^a-z0-9_]", "", safe)

    # Remove consecutive underscores (e.g. "a___b" → "a__b")
    safe = re.sub(r"_{3,}", "__", safe)

    # Strip leading/trailing underscores
    safe = safe.strip("_")

    # Truncate to max length
    safe = safe[:MAX_COLLECTION_NAME_LEN]

    # Pad if too short
    if len(safe) < MIN_COLLECTION_NAME_LEN:
        safe = safe.ljust(MIN_COLLECTION_NAME_LEN, "x")

    return safe


def get_or_create_collection(repo_name: str, persist_dir: str = None):
    """
    Open an existing ChromaDB collection or create a new one.

    One collection = one repository.
    The collection uses cosine distance for similarity comparison, which
    matches the normalized vectors produced by embedder.embed_text().

    Args:
        repo_name:   Repository identifier, e.g. "tiangolo__fastapi".
                     Will be sanitized automatically.
        persist_dir: ChromaDB storage directory. Optional.

    Returns:
        A chromadb.Collection object.
    """
    client     = get_client(persist_dir)
    safe_name  = sanitize_collection_name(repo_name)

    collection = client.get_or_create_collection(
        name     = safe_name,
        metadata = {"hnsw:space": "cosine"}  # use cosine similarity
    )
    logger.debug(f"Collection '{safe_name}': {collection.count()} chunks")
    return collection


def upsert_chunks(
    chunks:      list[dict],
    repo_name:   str,
    persist_dir: str = None,
) -> int:
    """
    Store embedded chunks in ChromaDB.

    Each chunk must have an 'embedding' field (list of 384 floats) added
    by embedder.embed_batch() before calling this function.

    The function uses "upsert" (update or insert) — if a chunk with the
    same ID already exists, it is updated. This makes re-indexing safe:
    running build_vector_index.py twice won't duplicate chunks.

    IDs are constructed as: "{file_path}::{chunk_name}::{start_line}"
    Example ID: "src/auth.py::UserService.login::42"

    Args:
        chunks:      List of chunk dicts, each with an 'embedding' key.
        repo_name:   Repository name (used as collection name).
        persist_dir: ChromaDB storage directory. Optional.

    Returns:
        Number of chunks upserted.

    Raises:
        ValueError: If any chunk is missing required fields or embedding.
    """
    if not chunks:
        logger.warning("upsert_chunks called with empty chunk list")
        return 0

    collection = get_or_create_collection(repo_name, persist_dir)

    ids         = []
    embeddings  = []
    documents   = []
    metadatas   = []

    for chunk in chunks:
        # Validate required fields
        required = {"text", "file_path", "chunk_name", "chunk_type",
                    "start_line", "end_line", "language"}
        missing = required - set(chunk.keys())
        if missing:
            logger.warning(f"Skipping chunk missing fields {missing}: {chunk.get('chunk_name','?')}")
            continue

        if "embedding" not in chunk or not chunk["embedding"]:
            logger.warning(f"Skipping chunk with no embedding: {chunk.get('chunk_name','?')}")
            continue

        # Build a unique ID for this chunk
        chunk_id = (
            f"{chunk['file_path']}::{chunk['chunk_name']}::{chunk['start_line']}"
        )

        # Metadata must be flat dict with scalar values (ChromaDB requirement)
        # Integers, floats, and strings are all supported
        meta = {
            "file_path":      str(chunk["file_path"]),
            "chunk_type":     str(chunk["chunk_type"]),
            "chunk_name":     str(chunk["chunk_name"]),
            "language":       str(chunk["language"]),
            "start_line":     int(chunk["start_line"]),
            "end_line":       int(chunk["end_line"]),
            "char_count":     int(chunk.get("char_count", len(chunk["text"]))),
            "token_estimate": int(chunk.get("token_estimate", len(chunk["text"]) // 4)),
            "repo_name":      str(chunk.get("repo_name", repo_name)),
        }

        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append(meta)

    if not ids:
        logger.warning("No valid chunks to upsert after validation")
        return 0

    # ChromaDB upsert in one batch call
    collection.upsert(
        ids        = ids,
        embeddings = embeddings,
        documents  = documents,
        metadatas  = metadatas,
    )

    count = len(ids)
    logger.info(f"Upserted {count} chunks into collection '{sanitize_collection_name(repo_name)}'")
    return count


def similarity_search(
    query_vector: list[float],
    repo_name:    str,
    top_k:        int = 5,
    persist_dir:  str = None,
    where:        dict = None,
) -> list[dict]:
    """
    Search ChromaDB for the most semantically similar chunks.

    The search computes cosine distance between the query vector and all
    stored chunk vectors, then returns the top_k most similar results.

    IMPORTANT: ChromaDB with cosine distance returns DISTANCE (0=identical,
    2=opposite). We convert this to SIMILARITY (1=identical, -1=opposite)
    using: similarity = 1 - distance.
    Higher similarity = better match.

    Args:
        query_vector: Embedding of the user's search query (384 floats).
        repo_name:    Repository name (collection to search in).
        top_k:        Number of results to return. Default: 5.
        persist_dir:  ChromaDB storage directory. Optional.
        where:        Optional metadata filter dict.
                      Example: {"language": "python"} to search only Python chunks.

    Returns:
        List of result dicts, sorted by similarity (highest first):
        [
            {
                "score":         0.87,
                "text":          "def login(user, password): ...",
                "file_path":     "src/auth.py",
                "chunk_name":    "login",
                "chunk_type":    "function",
                "start_line":    42,
                "end_line":      58,
                "language":      "python",
                "token_estimate":78,
            },
            ...
        ]

    Raises:
        RuntimeError: If the collection doesn't exist or search fails.
    """
    collection = get_or_create_collection(repo_name, persist_dir)

    count = collection.count()
    if count == 0:
        logger.warning(f"Collection '{repo_name}' is empty. Run build_vector_index.py first.")
        return []

    # Clamp top_k to available documents
    top_k = min(top_k, count)

    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results":        top_k,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        raise RuntimeError(f"ChromaDB search failed: {e}") from e

    output = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # Convert cosine distance → cosine similarity
        similarity = round(1.0 - dist, 4)

        result = {
            "score":          similarity,
            "text":           doc,
            "file_path":      meta.get("file_path", ""),
            "chunk_name":     meta.get("chunk_name", ""),
            "chunk_type":     meta.get("chunk_type", ""),
            "start_line":     meta.get("start_line", 0),
            "end_line":       meta.get("end_line",   0),
            "language":       meta.get("language",   ""),
            "char_count":     meta.get("char_count",     0),
            "token_estimate": meta.get("token_estimate", 0),
            "repo_name":      meta.get("repo_name",      ""),
        }
        output.append(result)

    return output


def collection_count(repo_name: str, persist_dir: str = None) -> int:
    """
    Return the number of chunks stored in a collection.

    Args:
        repo_name:   Repository name (collection name).
        persist_dir: ChromaDB storage directory. Optional.

    Returns:
        Integer count of chunks. Returns 0 if collection doesn't exist yet.
    """
    try:
        collection = get_or_create_collection(repo_name, persist_dir)
        return collection.count()
    except Exception as e:
        logger.warning(f"Could not count collection '{repo_name}': {e}")
        return 0


def delete_collection(repo_name: str, persist_dir: str = None) -> bool:
    """
    Delete a ChromaDB collection (and all its data).

    Use this before re-indexing to ensure a clean slate.

    Args:
        repo_name:   Repository name (collection to delete).
        persist_dir: ChromaDB storage directory. Optional.

    Returns:
        True if deleted, False if collection didn't exist.
    """
    client    = get_client(persist_dir)
    safe_name = sanitize_collection_name(repo_name)

    try:
        client.delete_collection(safe_name)
        logger.info(f"Deleted collection: {safe_name}")
        return True
    except Exception as e:
        logger.debug(f"Collection '{safe_name}' not found or already deleted: {e}")
        return False


def list_collections(persist_dir: str = None) -> list[str]:
    """
    List all collection names in the ChromaDB database.

    Args:
        persist_dir: ChromaDB storage directory. Optional.

    Returns:
        List of collection name strings.
    """
    client = get_client(persist_dir)
    try:
        collections = client.list_collections()
        return [c.name for c in collections]
    except Exception as e:
        logger.warning(f"Could not list collections: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST (requires chromadb installed)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("Testing core/vector_store.py\n" + "─" * 50)

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Test 1: Sanitize collection names
        print("\nTest 1: sanitize_collection_name()")
        assert sanitize_collection_name("tiangolo/fastapi")   == "tiangolo__fastapi"
        assert sanitize_collection_name("My-Org/My.Repo")     == "my_org__my_repo"
        assert sanitize_collection_name("psf/requests")       == "psf__requests"
        print("  ✓ All name sanitization tests pass")

        # Test 2: Create collection
        print("\nTest 2: get_or_create_collection()")
        col = get_or_create_collection("test__repo", tmp_dir)
        assert col is not None
        print(f"  ✓ Collection created: {col.name}")

        # Test 3: Upsert chunks
        print("\nTest 3: upsert_chunks()")
        test_chunks = [
            {
                "text":           "def login(user, password):\n    return check(user, password)",
                "file_path":      "src/auth.py",
                "chunk_name":     "login",
                "chunk_type":     "function",
                "language":       "python",
                "start_line":     10,
                "end_line":       12,
                "char_count":     60,
                "token_estimate": 15,
                "embedding":      [0.1] * 384,
            },
            {
                "text":           "class UserService:\n    def get_user(self, id): ...",
                "file_path":      "src/services.py",
                "chunk_name":     "UserService",
                "chunk_type":     "class",
                "language":       "python",
                "start_line":     1,
                "end_line":       20,
                "char_count":     200,
                "token_estimate": 50,
                "embedding":      [0.2] * 384,
            },
        ]
        n = upsert_chunks(test_chunks, "test__repo", tmp_dir)
        assert n == 2, f"Expected 2, got {n}"
        print(f"  ✓ Upserted {n} chunks")

        # Test 4: collection_count
        print("\nTest 4: collection_count()")
        count = collection_count("test__repo", tmp_dir)
        assert count == 2, f"Expected 2, got {count}"
        print(f"  ✓ Count: {count}")

        # Test 5: similarity_search
        print("\nTest 5: similarity_search()")
        # Use the same vector as the first chunk — should be returned first
        results = similarity_search([0.1] * 384, "test__repo", top_k=2, persist_dir=tmp_dir)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert "file_path" in results[0], "Result missing file_path"
        assert "score"     in results[0], "Result missing score"
        print(f"  ✓ Top result: {results[0]['chunk_name']} (score={results[0]['score']:.3f})")

        # Test 6: delete_collection
        print("\nTest 6: delete_collection()")
        deleted = delete_collection("test__repo", tmp_dir)
        assert deleted, "Should return True on successful delete"
        count_after = collection_count("test__repo", tmp_dir)
        assert count_after == 0, "Should be 0 after deletion"
        print("  ✓ Deletion successful")

    print("\n✓ All vector_store tests passed!")
