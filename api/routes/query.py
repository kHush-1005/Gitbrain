"""
api/routes/query.py  [NEW — Week 4]
─────────────────────────────────────────────────────────────────────────────
POST /query

Main API endpoint. Accepts a natural language question about a repository
and returns a cited AI-generated answer.

REQUEST BODY:
    {
        "question":  "how does user authentication work?",
        "repo_name": "psf__requests",
        "top_k":     5              (optional, default from settings)
    }

RESPONSE BODY:
    {
        "answer":  "The authentication in this repository is handled by...",
        "sources": [
            {"file": "requests/auth.py", "lines": "1-50",  "score": 0.89},
            {"file": "requests/utils.py", "lines": "45-80", "score": 0.71}
        ],
        "chunks_retrieved": 3,
        "repo_name": "psf__requests"
    }

ERROR RESPONSES:
    422 Unprocessable Entity — missing or invalid fields (Pydantic handles this)
    400 Bad Request         — empty question
    404 Not Found           — repository not indexed
    500 Internal Server Error — Groq or ChromaDB failure
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Request body for POST /query.

    Fields:
        question:  The natural language question to ask.
        repo_name: The repository collection name in ChromaDB.
                   Must match what was used in build_vector_index.py.
                   Example: "psf__requests", "tiangolo__fastapi"
        top_k:     Optional. How many chunks to retrieve (default: from settings).
    """
    question:  str            = Field(..., min_length=1, description="Natural language question")
    repo_name: str            = Field(..., min_length=1, description="ChromaDB collection name (e.g. psf__requests)")
    top_k:     Optional[int]  = Field(None,  ge=1, le=20, description="Chunks to retrieve (1-20)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question":  "how does user authentication work?",
                "repo_name": "psf__requests",
                "top_k":     5
            }
        }
    }


class SourceCitation(BaseModel):
    """A single source file citation."""
    file:  str   = Field(..., description="File path in the repository")
    lines: str   = Field(..., description="Line range, e.g. '42-58'")
    score: float = Field(..., description="Cosine similarity score (0.0-1.0)")


class QueryResponse(BaseModel):
    """
    Response body for POST /query.

    Fields:
        answer:           The LLM-generated answer with inline [FILE: ...] citations.
        sources:          Structured list of cited source files.
        chunks_retrieved: How many chunks passed the similarity threshold.
        repo_name:        The repository that was queried.
    """
    answer:           str               = Field(..., description="LLM-generated answer")
    sources:          list[SourceCitation] = Field(..., description="Cited source files")
    chunks_retrieved: int               = Field(..., description="Chunks above similarity threshold")
    repo_name:        str               = Field(..., description="Repository queried")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Authentication in this repository is handled by the HTTPBasicAuth class in requests/auth.py (Lines 1-50). [FILE: requests/auth.py, Lines 1-50]",
                "sources": [
                    {"file": "requests/auth.py", "lines": "1-50", "score": 0.89}
                ],
                "chunks_retrieved": 3,
                "repo_name": "psf__requests"
            }
        }
    }


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest) -> QueryResponse:
    """
    Ask a natural language question about an indexed GitHub repository.

    The system will:
    1. Convert your question to a vector embedding
    2. Search the ChromaDB index for the most relevant code chunks
    3. Build a prompt with those chunks as context
    4. Send the prompt to Llama 3 via Groq
    5. Return the answer with source file citations

    Prerequisites:
    - The repository must be indexed first using build_vector_index.py
    - GROQ_API_KEY must be set in your .env file

    Error cases:
    - 400: Empty question
    - 404: Repository not indexed (run build_vector_index.py first)
    - 503: Groq API unavailable (check console.groq.com)
    """
    from core.rag_engine import answer_question

    logger.info(
        f"POST /query | repo={request.repo_name!r} | "
        f"question={request.question[:60]!r}"
    )

    # ── Call the RAG pipeline ──────────────────────────────────────────────────
    try:
        result = answer_question(
            question  = request.question.strip(),
            repo_name = request.repo_name.strip(),
            top_k     = request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        error_msg = str(e)
        # Distinguish between "not indexed" and "API failure"
        if "No vector index" in error_msg or "empty" in error_msg.lower():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Repository '{request.repo_name}' is not indexed. "
                    f"Run: python scripts\\build_vector_index.py data\\chunks\\{request.repo_name}_chunks.jsonl"
                )
            )
        if "GROQ_API_KEY" in error_msg:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY is not configured. Add it to your .env file."
            )
        # Generic server error
        logger.error(f"RAG engine error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in /query: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    # ── Build response ─────────────────────────────────────────────────────────
    sources = [
        SourceCitation(
            file  = s["file"],
            lines = s["lines"],
            score = s["score"],
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        answer           = result["answer"],
        sources          = sources,
        chunks_retrieved = result["chunks_retrieved"],
        repo_name        = result["repo_name"],
    )
