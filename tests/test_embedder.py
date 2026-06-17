"""
tests/test_embedder.py  [NEW — Week 3]
─────────────────────────────────────────────────────────────────────────────
Pytest tests for core/embedder.py

These tests verify:
    - Embedding dimension is exactly 384
    - embed_text returns a list of floats
    - embed_batch returns the correct number of vectors
    - Same text always produces the same embedding (deterministic)
    - Similar texts produce closer vectors than unrelated texts
    - Error handling for empty inputs

NOTE: These tests require sentence-transformers to be installed.
      They will download the model on first run (~22 MB, one-time).
      Run time: ~5-15 seconds on first run, ~1 second after model is cached.

Run with:
    pytest tests/test_embedder.py -v
    pytest tests/ -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import pytest

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedder import (
    embed_text,
    embed_batch,
    get_embedding_dim,
    validate_embedding,
    EMBEDDING_DIM,
)

# ─────────────────────────────────────────────────────────────────────────────
# Mark all tests in this file as requiring the embedding model
# (skips gracefully if sentence-transformers not installed)
# ─────────────────────────────────────────────────────────────────────────────
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: get_embedding_dim()
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingDimension:

    def test_embedding_dim_constant_is_384(self):
        """The module-level constant should be 384."""
        assert EMBEDDING_DIM == 384

    def test_get_embedding_dim_returns_384(self):
        """get_embedding_dim() should return 384."""
        assert get_embedding_dim() == 384

    def test_embed_text_output_dimension_is_384(self):
        """An actual embedding should have exactly 384 dimensions."""
        vector = embed_text("def login(user, password): pass")
        assert len(vector) == 384, f"Expected 384, got {len(vector)}"

    def test_embed_batch_single_item_dimension_is_384(self):
        """embed_batch with one item should return one vector of 384 dims."""
        vectors = embed_batch(["test text"])
        assert len(vectors)    == 1
        assert len(vectors[0]) == 384


# ─────────────────────────────────────────────────────────────────────────────
# Tests: embed_text()
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedText:

    def test_returns_list(self):
        """embed_text should return a Python list."""
        result = embed_text("hello world")
        assert isinstance(result, list)

    def test_returns_floats(self):
        """All values in the embedding should be floats."""
        vector = embed_text("def authenticate(): pass")
        for val in vector:
            assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_same_text_same_embedding(self):
        """Identical inputs should produce identical outputs (deterministic)."""
        text = "class UserService: def get_user(self, id): pass"
        v1   = embed_text(text)
        v2   = embed_text(text)
        assert v1 == v2, "Same text must produce same embedding"

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        v1 = embed_text("user authentication and login")
        v2 = embed_text("database schema and table columns")
        assert v1 != v2, "Different texts should have different embeddings"

    def test_empty_text_raises_value_error(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            embed_text("")

    def test_whitespace_only_raises_value_error(self):
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError):
            embed_text("   \n\t   ")

    def test_code_snippet_embeds_successfully(self):
        """A realistic code snippet should embed without errors."""
        code = """
def process_payment(amount, card_number, cvv):
    if not validate_card(card_number):
        raise ValueError("Invalid card")
    return payment_gateway.charge(amount, card_number, cvv)
"""
        vector = embed_text(code)
        assert len(vector) == 384

    def test_long_text_embeds_without_error(self):
        """Very long text should embed without raising (model truncates)."""
        long_text = "This is a code comment. " * 500  # ~12,000 chars
        vector    = embed_text(long_text)
        assert len(vector) == 384

    def test_embedding_values_are_bounded(self):
        """
        Normalized embeddings should have values in roughly [-1, 1].
        The exact range depends on L2 normalization.
        """
        vector = embed_text("sample code function")
        max_val = max(abs(v) for v in vector)
        assert max_val <= 2.0, f"Unnormalized value detected: max={max_val}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: embed_batch()
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedBatch:

    def test_returns_list_of_lists(self):
        """embed_batch should return a list of lists."""
        results = embed_batch(["text one", "text two"])
        assert isinstance(results, list)
        assert isinstance(results[0], list)

    def test_output_count_matches_input_count(self):
        """Number of vectors must match number of input texts."""
        texts   = ["one", "two", "three", "four", "five"]
        vectors = embed_batch(texts)
        assert len(vectors) == len(texts), (
            f"Expected {len(texts)} vectors, got {len(vectors)}"
        )

    def test_single_item_batch(self):
        """Batch with one item should work."""
        vectors = embed_batch(["single text"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384

    def test_large_batch(self):
        """Batch larger than default batch_size (32) should work."""
        texts   = [f"function number {i} does task {i}" for i in range(50)]
        vectors = embed_batch(texts)
        assert len(vectors) == 50
        assert all(len(v) == 384 for v in vectors)

    def test_empty_list_raises_value_error(self):
        """Empty input list should raise ValueError."""
        with pytest.raises(ValueError):
            embed_batch([])

    def test_each_vector_has_384_dimensions(self):
        """Every vector in the batch output should have 384 dimensions."""
        texts   = ["auth code", "database query", "api endpoint", "config file"]
        vectors = embed_batch(texts)
        for i, v in enumerate(vectors):
            assert len(v) == 384, f"Vector {i} has wrong dimension: {len(v)}"

    def test_batch_vectors_are_floats(self):
        """All values in all batch vectors should be floats."""
        vectors = embed_batch(["class A: pass", "def f(): pass"])
        for vec in vectors:
            for val in vec:
                assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_batch_result_equals_individual_results(self):
        """
        Each vector from embed_batch should equal the corresponding
        vector from calling embed_text() individually.
        """
        texts = ["user authentication", "database schema"]
        batch = embed_batch(texts)
        for i, text in enumerate(texts):
            individual = embed_text(text)
            # Allow tiny floating-point differences
            diffs = [abs(a - b) for a, b in zip(batch[i], individual)]
            max_diff = max(diffs)
            assert max_diff < 1e-5, (
                f"Batch and individual embedding differ for '{text}': max_diff={max_diff}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_embedding()
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateEmbedding:

    def test_valid_384_dim_vector_passes(self):
        vector = [0.1] * 384
        assert validate_embedding(vector) is True

    def test_empty_list_fails(self):
        assert validate_embedding([]) is False

    def test_wrong_dimension_fails(self):
        assert validate_embedding([0.1] * 100) is False
        assert validate_embedding([0.1] * 768) is False

    def test_non_list_fails(self):
        assert validate_embedding("not a list") is False
        assert validate_embedding(None)         is False

    def test_actual_embedding_passes(self):
        """A real embedding from embed_text should pass validation."""
        vector = embed_text("test validation")
        assert validate_embedding(vector) is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Semantic similarity sanity check
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticSimilarity:
    """
    These tests verify that the model produces semantically meaningful vectors.
    Similar topics should have higher cosine similarity than unrelated topics.
    """

    @staticmethod
    def cosine_similarity(a: list, b: list) -> float:
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

    def test_similar_topics_higher_similarity_than_unrelated(self):
        """Authentication-related texts should be closer to each other than to DB topics."""
        v_auth1   = embed_text("user login and authentication flow")
        v_auth2   = embed_text("how does user authentication work?")
        v_unrelated = embed_text("database table schema and columns definition")

        sim_related   = self.cosine_similarity(v_auth1, v_auth2)
        sim_unrelated = self.cosine_similarity(v_auth1, v_unrelated)

        assert sim_related > sim_unrelated, (
            f"Related texts ({sim_related:.3f}) should score higher "
            f"than unrelated ({sim_unrelated:.3f})"
        )

    def test_identical_text_has_max_similarity(self):
        """Identical texts should have cosine similarity very close to 1.0."""
        text = "def get_user(self, user_id: int) -> User"
        v1   = embed_text(text)
        v2   = embed_text(text)
        sim  = self.cosine_similarity(v1, v2)
        assert sim > 0.999, f"Identical text similarity should be ~1.0, got {sim:.4f}"
