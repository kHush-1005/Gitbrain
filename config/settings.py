"""
config/settings.py  [NEW — Week 4]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Central configuration module for GitBrain.

    All environment variables are loaded once here via Pydantic BaseSettings
    and exposed as a typed settings object. Every other module imports
    `settings` from this file — nothing reads os.getenv() directly.

    This guarantees:
        - One place to change any setting
        - Type validation (int stays int, float stays float)
        - Clear error messages when required variables are missing
        - Easy to test (override settings in tests without touching .env)

USAGE:
    from config.settings import settings

    print(settings.groq_api_key)
    print(settings.top_k_results)

ENVIRONMENT FILE:
    Settings are loaded from the .env file in the project root.
    Copy .env.example to .env and fill in your values.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    All GitBrain configuration values.

    Each field maps to an environment variable of the same name (uppercase).
    Example: github_token → GITHUB_TOKEN in .env
    """

    # ── GitHub API ────────────────────────────────────────────────────────────
    github_token: str = ""
    """
    GitHub Personal Access Token.
    Without: 60 requests/hour  |  With: 5,000 requests/hour
    Get at: github.com/settings/tokens
    """

    # ── Groq LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    """
    Groq API key for Llama 3 inference.
    Get a FREE key at: console.groq.com
    Required for /query endpoint to return AI answers.
    """

    llm_model: str = "llama3-8b-8192"
    """
    Groq model identifier.
    Options: llama3-8b-8192 | llama3-70b-8192 | llama-3.1-8b-instant
    """

    llm_temperature: float = 0.1
    """
    LLM sampling temperature.
    0.1 = factual/deterministic (recommended for code Q&A)
    0.7 = creative/varied
    """

    llm_max_tokens: int = 1024
    """Maximum tokens in the LLM response."""

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    """
    sentence-transformers model for generating embeddings.
    all-MiniLM-L6-v2: 384 dims, ~22 MB, Apache 2.0 license.
    Downloaded automatically on first use (~22 MB, one-time).
    """

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "data/chroma_db"
    """
    Directory where ChromaDB stores the vector index on disk.
    Created automatically if it doesn't exist.
    """

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    top_k_results: int = 5
    """
    Number of chunks to retrieve from ChromaDB per query.
    Higher = more context for LLM but slower and more tokens used.
    """

    similarity_threshold: float = 0.35
    """
    Minimum cosine similarity score (0.0-1.0) for a chunk to be included.
    Chunks below this are discarded before being sent to the LLM.
    Lower value = more permissive (more chunks pass, but some may be noisy).
    Higher value = stricter (fewer chunks, but all highly relevant).
    """

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    """Host to bind the FastAPI server to."""

    api_port: int = 8000
    """Port to bind the FastAPI server to."""

    debug_mode: bool = False
    """Enable FastAPI debug mode (auto-reload, detailed errors)."""

    model_config = {
        "env_file":          ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive":    False,
        "extra":             "ignore",    # ignore unknown env vars
    }


# ── Singleton instance ────────────────────────────────────────────────────────
# Import this object in all other modules:
#   from config.settings import settings
settings = Settings()


# ── Validation helpers ────────────────────────────────────────────────────────
def require_groq_key() -> None:
    """
    Raise a clear error if GROQ_API_KEY is not configured.

    Called at the start of any function that calls the Groq API.
    Gives beginners a clear, actionable error message instead of a
    cryptic AuthenticationError from the groq library.
    """
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "\n"
            "To fix this:\n"
            "  1. Go to console.groq.com and sign up (free)\n"
            "  2. Create an API key\n"
            "  3. Open your .env file\n"
            "  4. Add: GROQ_API_KEY=gsk_yourkey\n"
            "  5. Restart the server\n"
        )


def require_chroma_index(repo_name: str) -> None:
    """
    Raise a clear error if the ChromaDB index for repo_name doesn't exist.

    Called before attempting a search query.
    """
    from core.vector_store import collection_count
    count = collection_count(repo_name)
    if count == 0:
        raise RuntimeError(
            f"No vector index found for repository: '{repo_name}'\n"
            "\n"
            "To fix this:\n"
            f"  1. Make sure you have a chunk file: data/chunks/{repo_name}_chunks.jsonl\n"
            f"  2. Run: python scripts\\build_vector_index.py data\\chunks\\{repo_name}_chunks.jsonl\n"
            f"  3. Then retry your query\n"
        )


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("GitBrain Configuration\n" + "─" * 40)
    print(f"  LLM Model           : {settings.llm_model}")
    print(f"  Embedding Model     : {settings.embedding_model}")
    print(f"  ChromaDB Dir        : {settings.chroma_persist_dir}")
    print(f"  Top-K Results       : {settings.top_k_results}")
    print(f"  Similarity Threshold: {settings.similarity_threshold}")
    print(f"  Temperature         : {settings.llm_temperature}")
    print(f"  Max Tokens          : {settings.llm_max_tokens}")
    print(f"  GitHub Token        : {'✓ set' if settings.github_token else '✗ not set'}")
    print(f"  Groq API Key        : {'✓ set' if settings.groq_api_key else '✗ not set (add to .env)'}")
    print()
    if not settings.groq_api_key:
        print("  ⚠  GROQ_API_KEY is empty. Add it to .env before using /query.")
    else:
        print("  ✓ All required settings are configured.")
