"""
api/main.py  [NEW — Week 4]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    FastAPI application entry point.

    This file:
        - Creates the FastAPI app instance
        - Registers all route routers
        - Configures CORS (so a browser frontend can call this API)
        - Sets up logging
        - Prints startup information

HOW TO RUN:
    uvicorn api.main:app --reload --port 8000

    --reload  = auto-restart when code changes (development only)
    --port    = port to listen on (default: 8000)

ENDPOINTS:
    GET  /health  → API status check
    POST /query   → Ask a question about an indexed repository
    GET  /docs    → Swagger UI (interactive API documentation)
    GET  /redoc   → ReDoc UI (alternative documentation)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.query  import router as query_router

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "GitBrain API",
    description = (
        "AI-Powered GitHub Repository Intelligence.\n\n"
        "Ask natural language questions about any indexed GitHub repository "
        "and receive cited, accurate answers powered by Llama 3 via Groq.\n\n"
        "**Prerequisites:**\n"
        "1. Run `python scripts\\build_vector_index.py data\\chunks\\REPO_chunks.jsonl`\n"
        "2. Set `GROQ_API_KEY` in your `.env` file\n"
        "3. Query with `POST /query`"
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ─── CORS Middleware ──────────────────────────────────────────────────────────
# Allows the Streamlit frontend (Week 5) to call this API from a browser.
# In production, replace ["*"] with your specific domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],    # Replace with frontend URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── Register Routers ─────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(query_router)

# ─── Startup Event ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    """Print startup information when the server starts."""
    from config.settings import settings
    logger.info("─" * 60)
    logger.info("GitBrain API starting...")
    logger.info(f"  LLM Model    : {settings.llm_model}")
    logger.info(f"  Embedding    : {settings.embedding_model}")
    logger.info(f"  ChromaDB Dir : {settings.chroma_persist_dir}")
    logger.info(f"  Top-K        : {settings.top_k_results}")
    logger.info(f"  Threshold    : {settings.similarity_threshold}")
    logger.info(f"  Groq Key     : {'✓ configured' if settings.groq_api_key else '✗ MISSING'}")
    logger.info("─" * 60)
    logger.info("Docs: http://127.0.0.1:8000/docs")
    logger.info("─" * 60)


# ─── Run directly (alternative to uvicorn CLI) ────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from config.settings import settings
    uvicorn.run(
        "api.main:app",
        host    = settings.api_host,
        port    = settings.api_port,
        reload  = True,
        log_level = "info",
    )
