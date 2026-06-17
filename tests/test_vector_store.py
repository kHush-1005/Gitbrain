"""
tests/test_vector_store.py  [NEW — Week 3]
─────────────────────────────────────────────────────────────────────────────
Pytest tests for core/vector_store.py

Tests cover:
    - Collection name sanitization
    - Creating and managing ChromaDB collections
    - Upserting chunks with embeddings
    - Collection count verification
    - Similarity search returns correct results
    - Metadata preservation
    - Deletion and cleanup

All tests use a temporary directory for ChromaDB storage so they don't
interfere with the real data/chroma_db/ folder.

Run with:
    pytest tests/test_vector_store.py -v
    pytest tests/ -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_store import (
    sanitize_collection_name,
    get_or_create_collection,
    upsert_chunks,
    similarity_search,
    collection_count,
    delete_collection,
    list_collections,
    get_client,
)

# ─── Fixture: fresh temp dir for each test class ──────────────────────────────
@pytest.fixture
def tmp_chroma(tmp_path):
    """
    Provide a temporary ChromaDB directory and reset the client singleton
    so each test gets a fresh database.
    """
    # Reset the module-level singleton so each test gets a fresh client
    import core.vector_store as vs
    vs._chroma_client      = None
    vs._chroma_persist_dir = None
    yield str(tmp_path)
    # Cleanup: reset again after test
    vs._chroma_client      = None
    vs._chroma_persist_dir = None


# ─── Sample chunk factory ──────────────────────────────────────────────────────
def make_chunk(
    text="def login(u, p): return check(u, p)",
    file_path="src/auth.py",
    chunk_name="login",
    chunk_type="function",
    language="python",
    start_line=10,
    end_line=12,
    embedding=None,
) -> dict:
    """Create a test chunk dict with an optional custom embedding."""
    return {
        "text":           text,
        "file_path":      file_path,
        "chunk_name":     chunk_name,
        "chunk_type":     chunk_type,
        "language":       language,
        "start_line":     start_line,
        "end_line":       end_line,
        "char_count":     len(text),
        "token_estimate": len(text) // 4,
        "embedding":      embedding or ([0.1] * 384),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests: sanitize_collection_name()
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizeCollectionName:

    def test_simple_name_unchanged(self):
        assert sanitize_collection_name("psf__requests") == "psf__requests"

    def test_slash_becomes_double_underscore(self):
        result = sanitize_collection_name("tiangolo/fastapi")
        assert "/" not in result
        assert "tiangolo" in result
        assert "fastapi"  in result

    def test_hyphens_become_underscores(self):
        result = sanitize_collection_name("my-org__my-repo")
        assert "-" not in result

    def test_dots_become_underscores(self):
        result = sanitize_collection_name("my.repo.v2")
        assert "." not in result

    def test_uppercase_lowercased(self):
        result = sanitize_collection_name("MyOrg__MyRepo")
        assert result == result.lower()

    def test_result_minimum_length_3(self):
        result = sanitize_collection_name("ab")
        assert len(result) >= 3

    def test_result_max_length_63(self):
        long_name = "a" * 100
        result    = sanitize_collection_name(long_name)
        assert len(result) <= 63

    def test_only_valid_chars_in_result(self):
        """Result should only contain lowercase letters, digits, underscores."""
        import re
        result = sanitize_collection_name("My-Org / My.Repo-V2!")
        assert re.match(r"^[a-z0-9_]+$", result), f"Invalid chars in: {result}"

    def test_known_conversions(self):
        assert sanitize_collection_name("tiangolo/fastapi") == "tiangolo__fastapi"
        assert sanitize_collection_name("psf/requests")     == "psf__requests"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: get_or_create_collection()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetOrCreateCollection:

    def test_creates_new_collection(self, tmp_chroma):
        col = get_or_create_collection("test__repo", tmp_chroma)
        assert col is not None

    def test_collection_starts_empty(self, tmp_chroma):
        col = get_or_create_collection("empty__repo", tmp_chroma)
        assert col.count() == 0

    def test_same_name_returns_same_collection(self, tmp_chroma):
        col1 = get_or_create_collection("same__repo", tmp_chroma)
        col2 = get_or_create_collection("same__repo", tmp_chroma)
        assert col1.name == col2.name

    def test_different_names_return_different_collections(self, tmp_chroma):
        col1 = get_or_create_collection("repo__one",   tmp_chroma)
        col2 = get_or_create_collection("repo__two",   tmp_chroma)
        assert col1.name != col2.name

    def test_name_is_sanitized(self, tmp_chroma):
        col = get_or_create_collection("My-Org/My-Repo", tmp_chroma)
        assert col.name == sanitize_collection_name("My-Org/My-Repo")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: upsert_chunks()
# ─────────────────────────────────────────────────────────────────────────────

class TestUpsertChunks:

    def test_upsert_single_chunk(self, tmp_chroma):
        chunk = make_chunk()
        n     = upsert_chunks([chunk], "test__repo", tmp_chroma)
        assert n == 1

    def test_upsert_multiple_chunks(self, tmp_chroma):
        chunks = [
            make_chunk(chunk_name="func_a", start_line=1,  end_line=5,  embedding=[0.1]*384),
            make_chunk(chunk_name="func_b", start_line=10, end_line=15, embedding=[0.2]*384),
            make_chunk(chunk_name="func_c", start_line=20, end_line=25, embedding=[0.3]*384),
        ]
        n = upsert_chunks(chunks, "test__repo", tmp_chroma)
        assert n == 3

    def test_upsert_updates_collection_count(self, tmp_chroma):
        chunks = [make_chunk(chunk_name=f"func_{i}", start_line=i, end_line=i+1) for i in range(5)]
        upsert_chunks(chunks, "count__test", tmp_chroma)
        assert collection_count("count__test", tmp_chroma) == 5

    def test_upsert_same_chunk_twice_no_duplicate(self, tmp_chroma):
        """Upserting the same chunk ID twice should update, not duplicate."""
        chunk = make_chunk(chunk_name="login", start_line=10)
        upsert_chunks([chunk], "dedup__test", tmp_chroma)
        upsert_chunks([chunk], "dedup__test", tmp_chroma)
        assert collection_count("dedup__test", tmp_chroma) == 1

    def test_upsert_empty_list_returns_zero(self, tmp_chroma):
        n = upsert_chunks([], "test__repo", tmp_chroma)
        assert n == 0

    def test_chunk_missing_embedding_is_skipped(self, tmp_chroma):
        chunk = make_chunk()
        del chunk["embedding"]
        n = upsert_chunks([chunk], "skip__test", tmp_chroma)
        assert n == 0

    def test_chunk_missing_required_field_is_skipped(self, tmp_chroma):
        chunk = make_chunk()
        del chunk["file_path"]
        n = upsert_chunks([chunk], "skip__test", tmp_chroma)
        assert n == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: collection_count()
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectionCount:

    def test_empty_collection_returns_zero(self, tmp_chroma):
        count = collection_count("empty__repo", tmp_chroma)
        assert count == 0

    def test_count_after_upsert(self, tmp_chroma):
        chunks = [make_chunk(chunk_name=f"f{i}", start_line=i, end_line=i+1) for i in range(7)]
        upsert_chunks(chunks, "count__repo", tmp_chroma)
        assert collection_count("count__repo", tmp_chroma) == 7

    def test_count_returns_integer(self, tmp_chroma):
        count = collection_count("new__repo", tmp_chroma)
        assert isinstance(count, int)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: similarity_search()
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilaritySearch:

    @pytest.fixture(autouse=True)
    def setup_test_collection(self, tmp_chroma):
        """Insert test chunks before each test in this class."""
        self.tmp = tmp_chroma
        # auth-related chunk: vector biased toward index 0-10
        auth_vec    = [0.9 if i < 10 else 0.0 for i in range(384)]
        # db-related chunk: vector biased toward index 20-30
        db_vec      = [0.0 if i < 20 else (0.9 if i < 30 else 0.0) for i in range(384)]
        # another auth chunk
        auth2_vec   = [0.85 if i < 10 else 0.0 for i in range(384)]

        chunks = [
            make_chunk(
                text="def login(user, password): return authenticate(user, password)",
                chunk_name="login", file_path="src/auth.py", start_line=10,
                embedding=auth_vec,
            ),
            make_chunk(
                text="CREATE TABLE users (id INT, name VARCHAR(255))",
                chunk_name="users_table", file_path="schema.sql",
                start_line=1, chunk_type="window", language="sql",
                embedding=db_vec,
            ),
            make_chunk(
                text="class AuthService: def verify_token(self, token): pass",
                chunk_name="AuthService", chunk_type="class",
                file_path="src/auth_service.py", start_line=5,
                embedding=auth2_vec,
            ),
        ]
        upsert_chunks(chunks, "search__test", tmp_chroma)

    def test_returns_list(self):
        results = similarity_search([0.1]*384, "search__test", top_k=3, persist_dir=self.tmp)
        assert isinstance(results, list)

    def test_returns_correct_number_of_results(self):
        results = similarity_search([0.1]*384, "search__test", top_k=2, persist_dir=self.tmp)
        assert len(results) == 2

    def test_results_have_required_fields(self):
        results = similarity_search([0.1]*384, "search__test", top_k=1, persist_dir=self.tmp)
        required = {"score","text","file_path","chunk_name","chunk_type",
                    "start_line","end_line","language"}
        for field in required:
            assert field in results[0], f"Missing field: {field}"

    def test_score_is_float_between_neg1_and_1(self):
        results = similarity_search([0.1]*384, "search__test", top_k=3, persist_dir=self.tmp)
        for r in results:
            assert isinstance(r["score"], float)
            assert -1.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_results_sorted_by_score_descending(self):
        results = similarity_search([0.1]*384, "search__test", top_k=3, persist_dir=self.tmp)
        scores  = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_auth_query_returns_auth_chunk_first(self):
        """
        A query vector biased toward auth (index 0-10) should return
        the auth chunk first, not the DB chunk.
        """
        auth_query = [0.9 if i < 10 else 0.0 for i in range(384)]
        results    = similarity_search(auth_query, "search__test", top_k=3, persist_dir=self.tmp)
        top_result = results[0]
        assert "auth" in top_result["file_path"].lower() or \
               "login" in top_result["chunk_name"].lower() or \
               "auth"  in top_result["chunk_name"].lower(), \
               f"Auth query should return auth chunk first, got: {top_result['chunk_name']}"

    def test_empty_collection_returns_empty_list(self, tmp_chroma):
        results = similarity_search([0.1]*384, "nonexistent__repo", top_k=5, persist_dir=tmp_chroma)
        assert results == []

    def test_top_k_clamped_to_collection_size(self):
        """Requesting more results than available should not raise an error."""
        results = similarity_search([0.1]*384, "search__test", top_k=100, persist_dir=self.tmp)
        # Collection has 3 chunks; should return at most 3
        assert len(results) <= 3

    def test_metadata_preserved_correctly(self):
        """Metadata stored with chunk should be returned with search results."""
        results = similarity_search([0.1]*384, "search__test", top_k=3, persist_dir=self.tmp)
        for r in results:
            assert isinstance(r["start_line"], int)
            assert isinstance(r["end_line"],   int)
            assert isinstance(r["language"],   str)
            assert r["start_line"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: delete_collection() and list_collections()
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteAndList:

    def test_delete_existing_collection_returns_true(self, tmp_chroma):
        get_or_create_collection("to__delete", tmp_chroma)
        result = delete_collection("to__delete", tmp_chroma)
        assert result is True

    def test_delete_nonexistent_collection_returns_false(self, tmp_chroma):
        result = delete_collection("does__not__exist", tmp_chroma)
        assert result is False

    def test_count_after_deletion_is_zero(self, tmp_chroma):
        chunk = make_chunk()
        upsert_chunks([chunk], "temp__repo", tmp_chroma)
        assert collection_count("temp__repo", tmp_chroma) == 1
        delete_collection("temp__repo", tmp_chroma)
        assert collection_count("temp__repo", tmp_chroma) == 0

    def test_list_collections_returns_list(self, tmp_chroma):
        result = list_collections(tmp_chroma)
        assert isinstance(result, list)

    def test_list_collections_includes_created_collection(self, tmp_chroma):
        get_or_create_collection("listed__repo", tmp_chroma)
        names = list_collections(tmp_chroma)
        safe  = sanitize_collection_name("listed__repo")
        assert safe in names
