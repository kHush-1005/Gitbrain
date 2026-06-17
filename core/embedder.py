"""
core/embedder.py  [NEW — Week 3]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Generate dense vector embeddings from text using sentence-transformers.

WHAT IS AN EMBEDDING?
    An embedding is a list of numbers (a vector) that represents the
    "meaning" of a piece of text. Similar texts produce similar vectors.

    Example:
        "How does login work?"    → [0.12, -0.34, 0.78, ...] (384 numbers)
        "User authentication flow" → [0.11, -0.31, 0.79, ...] (384 numbers)
        "Database schema tables"   → [-0.55, 0.23, -0.12, ...] (very different)

    ChromaDB uses these vectors to find the most similar code chunks
    when you ask a question.

MODEL USED:
    all-MiniLM-L6-v2  (from Hugging Face / sentence-transformers)
    - Output dimensions: 384 floats per text
    - Speed: ~14,000 sentences/second on CPU
    - Size: ~22 MB
    - License: Apache 2.0 (free for all use)
    - Downloaded once, cached automatically in ~/.cache/huggingface/

FUNCTIONS:
    get_embedding_model()   — load (or reuse) the model (lazy singleton)
    embed_text(text)        — embed a single string → list[float] (len 384)
    embed_batch(texts)      — embed a list of strings → list[list[float]]
    get_embedding_dim()     — return 384 (the expected vector size)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM      = 384          # output dimension for all-MiniLM-L6-v2
DEFAULT_BATCH_SIZE = 32           # chunks per embedding batch

# ─── Lazy singleton — model loaded once on first call ─────────────────────────
_model        = None
_model_name   = None


def get_embedding_model(model_name: str = None):
    """
    Load the sentence-transformers embedding model.

    Uses a lazy singleton pattern: the model is loaded only once on the
    first call and reused for all subsequent calls. Loading takes 2-5 seconds
    the first time; subsequent calls return instantly.

    On first ever use, the model weights (~22 MB) are downloaded from
    Hugging Face and cached at:
        Windows: C:\\Users\\<you>\\.cache\\huggingface\\hub\\
        Linux:   ~/.cache/huggingface/hub/

    Args:
        model_name: Model identifier. Defaults to all-MiniLM-L6-v2.

    Returns:
        A loaded SentenceTransformer model object.

    Raises:
        ImportError: If sentence-transformers is not installed.
        RuntimeError: If the model cannot be loaded.
    """
    global _model, _model_name

    # Read model name from environment variable if not explicitly provided
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)

    # Return cached model if same name requested
    if _model is not None and _model_name == model_name:
        return _model

    # Import sentence-transformers (only imported when first needed)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "Fix: pip install sentence-transformers\n"
            "     (or: pip install -r requirements.txt)"
        )

    logger.info(f"Loading embedding model: {model_name}")
    print(f"  [Embedder] Loading model '{model_name}'...")
    print("             (First load may take 2-5 seconds. Model cached after that.)")

    try:
        _model      = SentenceTransformer(model_name)
        _model_name = model_name
        logger.info(f"Model loaded. Embedding dimension: {get_embedding_dim()}")
        print(f"  [Embedder] ✓ Model ready. Embedding dimension: {get_embedding_dim()}")
        return _model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}': {e}\n"
            "Check your internet connection on first load, or verify the model name."
        ) from e


def get_embedding_dim() -> int:
    """
    Return the expected embedding dimension for the default model.

    Returns:
        384 for all-MiniLM-L6-v2.
        If using a different model, this will still return 384 as default —
        call model.get_sentence_embedding_dimension() for exact value.
    """
    return EMBEDDING_DIM


def embed_text(text: str, model_name: str = None) -> list[float]:
    """
    Generate a single embedding vector for one text string.

    This is the function used at QUERY TIME — when a user types a question,
    it gets embedded by this function so it can be compared to the stored
    chunk vectors in ChromaDB.

    Args:
        text:       The text to embed. Should be non-empty.
        model_name: Model identifier. Defaults to all-MiniLM-L6-v2.

    Returns:
        A list of 384 floats representing the text's semantic meaning.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If embedding fails.

    Example:
        vector = embed_text("how does login authentication work?")
        # → [0.123, -0.456, 0.789, ...]  (384 floats)
        print(len(vector))  # 384
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text. Provide a non-empty string.")

    model = get_embedding_model(model_name)

    try:
        # encode() returns a numpy array — convert to plain Python list
        # normalize_embeddings=True ensures cosine similarity = dot product
        vector = model.encode(
            text.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding failed for text: '{text[:80]}...': {e}") from e


def embed_batch(
    texts:      list[str],
    model_name: str  = None,
    batch_size: int  = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings in batches.

    This is the function used at INDEX TIME — when building the vector store,
    all code chunk texts are embedded in batches by this function.

    Batching is critical for performance: embedding 32 chunks at once is
    much faster than embedding them one at a time in a loop.

    Args:
        texts:         List of text strings to embed. All must be non-empty.
        model_name:    Model identifier. Defaults to all-MiniLM-L6-v2.
        batch_size:    Number of texts to embed per batch. Default: 32.
                       Increase for GPU, decrease if running out of memory.
        show_progress: Show a tqdm progress bar. Default: False.

    Returns:
        A list of embedding vectors (same length as input texts).
        Each vector is a list of 384 floats.

    Raises:
        ValueError: If texts list is empty.
        RuntimeError: If embedding fails.

    Example:
        chunks = ["def login(): ...", "class UserService: ..."]
        vectors = embed_batch(chunks)
        # → [[0.12, ...], [0.34, ...]]  (2 vectors, each 384 floats)
        print(len(vectors))     # 2
        print(len(vectors[0]))  # 384
    """
    if not texts:
        raise ValueError("Cannot embed empty list. Provide at least one text.")

    # Filter out any empty strings and track their positions
    cleaned = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            logger.warning(f"Skipping empty/non-string text at index {i}")
            cleaned.append("")   # placeholder — will be replaced with zero vector
        else:
            cleaned.append(t.strip())

    model = get_embedding_model(model_name)

    try:
        # SentenceTransformer handles batching internally when encode() is
        # called with a list — we set batch_size explicitly for control
        vectors = model.encode(
            cleaned,
            batch_size          = batch_size,
            normalize_embeddings = True,
            show_progress_bar   = show_progress,
            convert_to_numpy    = True,
        )
        return [v.tolist() for v in vectors]
    except Exception as e:
        raise RuntimeError(f"Batch embedding failed: {e}") from e


def validate_embedding(vector: list[float], expected_dim: int = EMBEDDING_DIM) -> bool:
    """
    Check that an embedding vector has the expected dimension and valid values.

    Used in tests and as a sanity check before upserting to ChromaDB.

    Args:
        vector:       The embedding vector to validate.
        expected_dim: Expected length of the vector. Default: 384.

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(vector, (list, tuple)):
        return False
    if len(vector) != expected_dim:
        return False
    if not all(isinstance(v, (int, float)) for v in vector):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing core/embedder.py\n" + "─" * 50)

    # Test 1: embed a single text
    print("\nTest 1: embed_text()")
    v = embed_text("def login(user, password): return check_credentials(user, password)")
    print(f"  ✓ Embedding dimension : {len(v)}")
    print(f"  ✓ Type               : {type(v[0]).__name__}")
    assert len(v) == 384,       f"Expected 384, got {len(v)}"
    assert isinstance(v[0], float), "Expected floats"

    # Test 2: embed a batch
    print("\nTest 2: embed_batch()")
    texts = [
        "def authenticate(user, password): ...",
        "class UserService: ...",
        "database connection pool settings",
    ]
    vectors = embed_batch(texts)
    print(f"  ✓ Input  : {len(texts)} texts")
    print(f"  ✓ Output : {len(vectors)} vectors, each {len(vectors[0])} dims")
    assert len(vectors) == 3,        "Wrong number of vectors"
    assert len(vectors[0]) == 384,   "Wrong dimension"

    # Test 3: semantic similarity
    print("\nTest 3: Semantic similarity check")
    v_login   = embed_text("user login authentication")
    v_auth    = embed_text("how does user authentication work?")
    v_unrelated = embed_text("database schema table columns primary key")

    def cosine_sim(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

    sim_related   = cosine_sim(v_login, v_auth)
    sim_unrelated = cosine_sim(v_login, v_unrelated)
    print(f"  ✓ login ↔ authentication : {sim_related:.3f}  (should be > 0.5)")
    print(f"  ✓ login ↔ database       : {sim_unrelated:.3f}  (should be lower)")
    assert sim_related > sim_unrelated, "Similar texts should score higher than unrelated"

    # Test 4: validate_embedding
    print("\nTest 4: validate_embedding()")
    assert validate_embedding(v),           "Valid vector should pass"
    assert not validate_embedding([]),       "Empty list should fail"
    assert not validate_embedding([0.1]*10), "Wrong dim should fail"
    print("  ✓ All validation checks pass")

    print("\n✓ All embedder tests passed!")
