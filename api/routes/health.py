"""
api/routes/health.py  [NEW — Week 4]
─────────────────────────────────────────────────────────────────────────────
GET /health

Returns a health check response confirming the API is running.
Also checks whether ChromaDB and Groq are accessible.

Used for:
    - Verifying the server started successfully
    - Monitoring in production
    - GitHub Actions automation checks (Week 6)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status:   str
    service:  str
    groq_key: str
    chroma:   str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the API and its dependencies.
    Always returns HTTP 200 — dependency status is shown in the body.

    Example response:
        {
            "status":   "ok",
            "service":  "GitBrain API",
            "groq_key": "configured",
            "chroma":   "data/chroma_db"
        }
    """
    from config.settings import settings

    groq_status  = "configured" if settings.groq_api_key else "MISSING — add GROQ_API_KEY to .env"
    chroma_status = settings.chroma_persist_dir

    logger.debug("Health check called")

    return HealthResponse(
        status   = "ok",
        service  = "GitBrain API",
        groq_key = groq_status,
        chroma   = chroma_status,
    )
