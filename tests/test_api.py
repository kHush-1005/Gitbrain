"""
tests/test_api.py  [NEW — Week 4]
─────────────────────────────────────────────────────────────────────────────
FastAPI endpoint tests using httpx TestClient.

Tests cover:
    - GET /health returns 200 and correct structure
    - POST /query validates required fields
    - POST /query returns 422 when question is missing
    - POST /query returns 422 when repo_name is missing
    - POST /query structure (mocked RAG engine)
    - POST /query returns 404 for unindexed repo

Run with:
    pytest tests/test_api.py -v

These tests do NOT require:
    - A real Groq API key
    - A real ChromaDB index
    - The embedding model
They use mocking to isolate the API layer.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ─── Shared mock response from the RAG engine ─────────────────────────────────
MOCK_RAG_RESPONSE = {
    "answer":           "The login function in src/auth.py handles authentication [FILE: src/auth.py, Lines 10-20].",
    "sources":          [{"file": "src/auth.py", "lines": "10-20", "score": 0.87}],
    "chunks_retrieved": 2,
    "repo_name":        "test__repo",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests: GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = client.get("/health")
        data     = response.json()
        assert data["status"] == "ok"

    def test_health_returns_service_name(self):
        response = client.get("/health")
        data     = response.json()
        assert data["service"] == "GitBrain API"

    def test_health_response_has_required_fields(self):
        response = client.get("/health")
        data     = response.json()
        assert "status"   in data
        assert "service"  in data
        assert "groq_key" in data
        assert "chroma"   in data

    def test_health_content_type_is_json(self):
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: POST /query — Input Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryValidation:

    def test_missing_question_returns_422(self):
        """Omitting 'question' field should return 422 Unprocessable Entity."""
        response = client.post("/query", json={"repo_name": "test__repo"})
        assert response.status_code == 422

    def test_missing_repo_name_returns_422(self):
        """Omitting 'repo_name' field should return 422."""
        response = client.post("/query", json={"question": "how does login work?"})
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_empty_question_returns_422(self):
        """Empty string question should fail Pydantic min_length=1 validation."""
        response = client.post("/query", json={"question": "", "repo_name": "test__repo"})
        assert response.status_code == 422

    def test_empty_repo_name_returns_422(self):
        response = client.post("/query", json={"question": "test?", "repo_name": ""})
        assert response.status_code == 422

    def test_top_k_zero_returns_422(self):
        """top_k has ge=1 constraint."""
        response = client.post(
            "/query",
            json={"question": "test?", "repo_name": "test__repo", "top_k": 0}
        )
        assert response.status_code == 422

    def test_top_k_too_large_returns_422(self):
        """top_k has le=20 constraint."""
        response = client.post(
            "/query",
            json={"question": "test?", "repo_name": "test__repo", "top_k": 99}
        )
        assert response.status_code == 422

    def test_valid_request_does_not_return_422(self):
        """A structurally valid request should not return 422 (even if RAG fails)."""
        with patch("core.rag_engine.answer_question") as mock_rag:
            mock_rag.side_effect = RuntimeError("No vector index found for repository")
            response = client.post(
                "/query",
                json={"question": "how does login work?", "repo_name": "test__repo"}
            )
        # 404 (not indexed) is acceptable — just not 422 (validation error)
        assert response.status_code != 422


# ─────────────────────────────────────────────────────────────────────────────
# Tests: POST /query — Response Structure (mocked RAG engine)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryResponse:

    def test_successful_query_returns_200(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG_RESPONSE):
            response = client.post(
                "/query",
                json={"question": "how does login work?", "repo_name": "test__repo"}
            )
        assert response.status_code == 200

    def test_response_has_answer_field(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG_RESPONSE):
            response = client.post(
                "/query",
                json={"question": "how does login work?", "repo_name": "test__repo"}
            )
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_response_has_sources_field(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG_RESPONSE):
            response = client.post(
                "/query",
                json={"question": "how does login work?", "repo_name": "test__repo"}
            )
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_response_has_chunks_retrieved_field(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG_RESPONSE):
            response = client.post(
                "/query",
                json={"question": "how does login work?", "repo_name": "test__repo"}
            )
        data = response.json()
        assert "chunks_retrieved" in data
        assert isinstance(data["chunks_retrieved"], int)

    def test_response_has_repo_name_field(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG_RESPONSE):
            response = client.post(
                "/query",
                json={"question": "how does login work?", "repo_name": "test__repo"}
            )
        data = response.json()
        assert "repo_name" in data

    def test_source_has_required_fields(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG_RESPONSE):
            response = client.post(
                "/query",
                json={"question": "how does login work?", "repo_name": "test__repo"}
            )
        data    = response.json()
        sources = data["sources"]
        if sources:
            source = sources[0]
            assert "file"  in source
            assert "lines" in source
            assert "score" in source

    def test_source_score_is_float(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG_RESPONSE):
            response = client.post(
                "/query",
                json={"question": "how does login work?", "repo_name": "test__repo"}
            )
        data = response.json()
        for source in data["sources"]:
            assert isinstance(source["score"], float)

    def test_custom_top_k_is_passed_to_rag(self):
        """top_k parameter should be forwarded to the RAG engine."""
        captured_kwargs = {}

        def mock_answer(**kwargs):
            captured_kwargs.update(kwargs)
            return MOCK_RAG_RESPONSE

        with patch("core.rag_engine.answer_question", side_effect=mock_answer):
            client.post(
                "/query",
                json={"question": "test?", "repo_name": "test__repo", "top_k": 3}
            )
        assert captured_kwargs.get("top_k") == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: POST /query — Error Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryErrorHandling:

    def test_unindexed_repo_returns_404(self):
        """RAG engine raising 'No vector index' should map to HTTP 404."""
        with patch(
            "core.rag_engine.answer_question",
            side_effect=RuntimeError("No vector index found for repository: 'unknown__repo'")
        ):
            response = client.post(
                "/query",
                json={"question": "test?", "repo_name": "unknown__repo"}
            )
        assert response.status_code == 404

    def test_missing_groq_key_returns_503(self):
        """Missing GROQ_API_KEY should return 503 Service Unavailable."""
        with patch(
            "core.rag_engine.answer_question",
            side_effect=RuntimeError("GROQ_API_KEY is not set.")
        ):
            response = client.post(
                "/query",
                json={"question": "test?", "repo_name": "test__repo"}
            )
        assert response.status_code == 503

    def test_empty_question_after_strip_returns_400(self):
        """If the question becomes empty after stripping, return 400."""
        with patch(
            "core.rag_engine.answer_question",
            side_effect=ValueError("Question cannot be empty.")
        ):
            response = client.post(
                "/query",
                json={"question": "  ", "repo_name": "test__repo"}
            )
        # Either 400 (from ValueError handler) or 422 (from Pydantic min_length)
        assert response.status_code in (400, 422)

    def test_no_relevant_context_returns_200_with_fallback(self):
        """
        When the RAG engine finds no relevant chunks, it returns the fallback
        message (not an error). The API should return 200 with that message.
        """
        fallback_response = {
            "answer":           "I could not find relevant information about this in the repository.",
            "sources":          [],
            "chunks_retrieved": 0,
            "repo_name":        "test__repo",
        }
        with patch("core.rag_engine.answer_question", return_value=fallback_response):
            response = client.post(
                "/query",
                json={"question": "capital of France?", "repo_name": "test__repo"}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_retrieved"] == 0
        assert "could not find" in data["answer"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: API Documentation
# ─────────────────────────────────────────────────────────────────────────────

class TestApiDocs:

    def test_swagger_docs_accessible(self):
        """GET /docs should return 200 (Swagger UI)."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self):
        """GET /redoc should return 200 (ReDoc UI)."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json_accessible(self):
        """GET /openapi.json should return the OpenAPI schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths"   in data
        assert "/health" in data["paths"]
        assert "/query"  in data["paths"]
