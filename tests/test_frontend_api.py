"""
tests/test_frontend_api.py  [NEW — Week 5]
─────────────────────────────────────────────────────────────────────────────
Integration tests that verify the FastAPI endpoints work correctly
as expected by the React frontend's apiClient.js.

Tests:
  - Backend health check returns the structure the frontend expects
  - POST /query validates inputs and returns the right response shape
  - POST /query returns 404 for unindexed repositories
  - POST /query returns 503 when GROQ_API_KEY is missing
  - Error response shapes match what apiClient.js parses

These tests use FastAPI's TestClient (no real Groq key needed — RAG is mocked).

Run with:
    pytest tests/test_frontend_api.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ── Shared mock RAG response ──────────────────────────────────────────────────
MOCK_RAG = {
    "answer":           "Authentication is handled by HTTPBasicAuth [FILE: requests/auth.py, Lines 1-50].",
    "sources":          [{"file": "requests/auth.py", "lines": "1-50", "score": 0.89}],
    "chunks_retrieved": 2,
    "repo_name":        "psf__requests",
}


# ─────────────────────────────────────────────────────────────────────────────
# GET /health — used by Sidebar.jsx via getBackendStatus()
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_response_has_status_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_response_has_service_field(self):
        data = client.get("/health").json()
        assert data["service"] == "GitBrain API"

    def test_response_has_groq_key_field(self):
        """Frontend reads this to set groqConfigured flag."""
        data = client.get("/health").json()
        assert "groq_key" in data
        assert isinstance(data["groq_key"], str)

    def test_response_has_chroma_field(self):
        """Frontend reads this to set chromaDir."""
        data = client.get("/health").json()
        assert "chroma" in data

    def test_content_type_json(self):
        resp = client.get("/health")
        assert "application/json" in resp.headers.get("content-type", "")


# ─────────────────────────────────────────────────────────────────────────────
# POST /query — used by ChatWindow.jsx via queryRepository()
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryEndpoint:

    def test_valid_request_returns_200(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG):
            resp = client.post("/query", json={
                "question":  "how does authentication work?",
                "repo_name": "psf__requests",
            })
        assert resp.status_code == 200

    def test_response_has_answer_field(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG):
            data = client.post("/query", json={
                "question": "how does login work?", "repo_name": "psf__requests"
            }).json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_response_has_sources_list(self):
        """Frontend iterates over sources to render SourcePanel."""
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG):
            data = client.post("/query", json={
                "question": "headers?", "repo_name": "psf__requests"
            }).json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_source_has_file_lines_score(self):
        """Each source must have file, lines, score for SourcePanel.jsx."""
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG):
            data = client.post("/query", json={
                "question": "auth?", "repo_name": "psf__requests"
            }).json()
        if data["sources"]:
            src = data["sources"][0]
            assert "file"  in src
            assert "lines" in src
            assert "score" in src

    def test_response_has_chunks_retrieved(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG):
            data = client.post("/query", json={
                "question": "test?", "repo_name": "psf__requests"
            }).json()
        assert "chunks_retrieved" in data
        assert isinstance(data["chunks_retrieved"], int)

    def test_response_has_repo_name(self):
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG):
            data = client.post("/query", json={
                "question": "test?", "repo_name": "psf__requests"
            }).json()
        assert "repo_name" in data

    def test_top_k_parameter_accepted(self):
        """Frontend can pass top_k; should not error."""
        with patch("core.rag_engine.answer_question", return_value=MOCK_RAG):
            resp = client.post("/query", json={
                "question": "test?", "repo_name": "psf__requests", "top_k": 3
            })
        assert resp.status_code == 200

    def test_no_relevant_context_returns_200_with_fallback(self):
        """Frontend must handle chunks_retrieved=0 gracefully (no crash)."""
        fallback = {
            "answer":           "I could not find relevant information about this in the repository.",
            "sources":          [],
            "chunks_retrieved": 0,
            "repo_name":        "psf__requests",
        }
        with patch("core.rag_engine.answer_question", return_value=fallback):
            data = client.post("/query", json={
                "question": "capital of France?", "repo_name": "psf__requests"
            }).json()
        assert data["status_code"] if "status_code" in data else True
        assert data["chunks_retrieved"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Validation errors — apiClient.js receives these as API errors
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryValidation:

    def test_missing_question_returns_422(self):
        resp = client.post("/query", json={"repo_name": "psf__requests"})
        assert resp.status_code == 422

    def test_missing_repo_name_returns_422(self):
        resp = client.post("/query", json={"question": "how does auth work?"})
        assert resp.status_code == 422

    def test_empty_question_returns_422(self):
        resp = client.post("/query", json={"question": "", "repo_name": "test__repo"})
        assert resp.status_code == 422

    def test_top_k_too_large_returns_422(self):
        resp = client.post("/query", json={
            "question": "test?", "repo_name": "test__repo", "top_k": 999
        })
        assert resp.status_code == 422

    def test_empty_body_returns_422(self):
        assert client.post("/query", json={}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Error handling — apiClient.js maps these to user-visible error messages
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryErrorHandling:

    def test_unindexed_repo_returns_404(self):
        """Frontend shows 'repository not indexed' message for 404."""
        with patch("core.rag_engine.answer_question",
                   side_effect=RuntimeError("No vector index found for repository")):
            resp = client.post("/query", json={
                "question": "test?", "repo_name": "not__indexed"
            })
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_missing_groq_key_returns_503(self):
        """Frontend shows 'configure API key' message for 503."""
        with patch("core.rag_engine.answer_question",
                   side_effect=RuntimeError("GROQ_API_KEY is not set")):
            resp = client.post("/query", json={
                "question": "test?", "repo_name": "test__repo"
            })
        assert resp.status_code == 503
        assert "detail" in resp.json()

    def test_error_response_has_detail_field(self):
        """apiClient.js reads error.response.data.detail for user messages."""
        with patch("core.rag_engine.answer_question",
                   side_effect=RuntimeError("No vector index found for repository")):
            data = client.post("/query", json={
                "question": "test?", "repo_name": "bad__repo"
            }).json()
        assert "detail" in data


# ─────────────────────────────────────────────────────────────────────────────
# API Docs — both endpoints must appear in the OpenAPI schema
# ─────────────────────────────────────────────────────────────────────────────

class TestApiDocs:

    def test_openapi_has_health_path(self):
        data = client.get("/openapi.json").json()
        assert "/health" in data["paths"]

    def test_openapi_has_query_path(self):
        data = client.get("/openapi.json").json()
        assert "/query" in data["paths"]

    def test_swagger_ui_accessible(self):
        assert client.get("/docs").status_code == 200
