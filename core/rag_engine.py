"""
core/rag_engine.py  [NEW — Week 4]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline.

    RAG = find relevant code → ask the LLM to explain it

WHAT THIS FILE DOES:
    1. Takes a user question
    2. Converts it to a vector embedding
    3. Searches ChromaDB for the most similar code chunks
    4. Filters out low-quality matches
    5. Builds a structured prompt with the retrieved code
    6. Sends the prompt to Llama 3 via Groq
    7. Parses the LLM output to extract source citations
    8. Returns the answer and source list

WHY RAG PREVENTS HALLUCINATION:
    The system prompt explicitly tells the LLM:
      "Answer ONLY from the code context provided below."
      "If the answer is not in the context, say so."
    This prevents the LLM from making up code that doesn't exist.

FUNCTIONS:
    answer_question(question, repo_name, top_k)  — main entry point
    build_prompt(question, chunks)               — assemble the LLM prompt
    parse_citations(llm_text, chunks)            — extract [FILE: ...] markers
─────────────────────────────────────────────────────────────────────────────
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── System prompt template ───────────────────────────────────────────────────
# This is the instruction set given to the LLM. It governs tone, citation
# format, and fallback behavior. Carefully worded to reduce hallucinations.
SYSTEM_PROMPT = """You are GitBrain, an expert AI code assistant that helps developers understand GitHub repositories.

Your rules:
1. Answer the user's question using ONLY the code context provided below.
2. For every claim you make, cite the source file using this exact format: [FILE: path/to/file.py, Lines X-Y]
3. If the answer cannot be found in the provided context, respond with exactly:
   "I could not find relevant information about this in the repository."
4. Do NOT invent, assume, or hallucinate any code, functions, or behavior not present in the context.
5. Be concise but complete. Use numbered steps for processes. Use inline code formatting for function names.
6. If a question spans multiple files, explain how they connect."""

# ─── Fallback message ─────────────────────────────────────────────────────────
NO_CONTEXT_ANSWER = "I could not find relevant information about this in the repository."

# Groq free tier allows ~6k tokens per request — keep context small.
MAX_CONTEXT_CHUNKS = 3
MAX_CHUNK_CHARS    = 1500
MAX_IPYNB_CHARS    = 800
MAX_PROMPT_CHARS   = 12000   # ~3k tokens — safe margin under 6k TPM limit


def _chunk_char_limit(chunk: dict) -> int:
    path = chunk.get("file_path", "").lower()
    if path.endswith(".ipynb"):
        return MAX_IPYNB_CHARS
    return MAX_CHUNK_CHARS


def _is_boilerplate_chunk(chunk: dict) -> bool:
    """Skip license/funding files — they are huge and rarely answer code questions."""
    path = chunk.get("file_path", "").replace("\\", "/").upper()
    if path == "LICENSE" or path.endswith("/LICENSE"):
        return True
    if path.endswith("FUNDING.YML") or path.endswith("/.GITHUB/FUNDING.YML"):
        return True
    return False


def _truncate_chunk_text(chunk: dict, max_chars: int | None = None) -> dict:
    if max_chars is None:
        max_chars = _chunk_char_limit(chunk)
    text = chunk.get("text", "")
    if len(text) <= max_chars:
        return chunk
    return {**chunk, "text": text[:max_chars] + "\n... [truncated]"}


def _prepare_chunks_for_llm(
    chunks: list[dict],
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    """
    Select a small set of chunks for the LLM prompt.

    Excludes boilerplate files, truncates long windows, and caps count so
    requests stay within Groq token limits.
    """
    prepared: list[dict] = []
    for chunk in chunks:
        if _is_boilerplate_chunk(chunk):
            continue
        prepared.append(_truncate_chunk_text(chunk, max_chars))
        if len(prepared) >= max_chunks:
            break

    if prepared:
        return prepared

    # Fallback: no useful chunks after filtering — use best non-boilerplate
    # matches only (never send raw LICENSE walls to the LLM).
    for chunk in chunks:
        if not _is_boilerplate_chunk(chunk):
            prepared.append(_truncate_chunk_text(chunk, max_chars))
            if len(prepared) >= max_chunks:
                break
    return prepared


def _prioritize_chunks(chunks: list[dict]) -> list[dict]:
    """
    Re-order retrieved chunks so overview docs (README) rank above boilerplate
    (LICENSE) when building the LLM context.
    """
    def sort_key(chunk: dict) -> tuple:
        path = chunk.get("file_path", "").replace("\\", "/").upper()
        score = chunk.get("score", 0)
        if path.endswith("README.MD") or path == "README.MD":
            return (0, -score)
        if path == "LICENSE" or path.endswith("/LICENSE"):
            return (2, -score)
        return (1, -score)

    return sorted(chunks, key=sort_key)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the full prompt that will be sent to the LLM.

    The prompt has three sections:
        1. SYSTEM_PROMPT — instructions for the LLM
        2. CODE CONTEXT  — the retrieved code chunks with file metadata
        3. USER QUESTION — the question to answer

    Each chunk is formatted as:
        --- Source N: path/to/file.py (Lines X-Y, Type: function) ---
        <chunk text>

    This format makes it easy for the LLM to reference specific files
    and for parse_citations() to extract those references.

    Args:
        question: The user's natural language question.
        chunks:   List of retrieved chunk dicts from similarity_search().
                  Each must have: text, file_path, start_line, end_line,
                  chunk_type, chunk_name, score.

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    if not chunks:
        # No context available — the caller should handle this before calling
        # build_prompt(), but we return a safe fallback prompt just in case
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"CODE CONTEXT:\nNo relevant code was found.\n\n"
            f"USER QUESTION: {question}\n\nANSWER:"
        )

    # Build context blocks
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"--- Source {i}: {chunk['file_path']} "
            f"(Lines {chunk['start_line']}-{chunk['end_line']}, "
            f"Type: {chunk['chunk_type']}) ---"
        )
        block = f"{header}\n{chunk['text']}"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CODE CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    if len(prompt) > MAX_PROMPT_CHARS:
        logger.warning(
            f"Prompt too large ({len(prompt)} chars) — trimming to {MAX_PROMPT_CHARS}"
        )
        overflow = len(prompt) - MAX_PROMPT_CHARS
        trimmed_context = context[:-overflow - 20] + "\n... [context truncated]"
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"CODE CONTEXT:\n{trimmed_context}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"ANSWER:"
        )

    return prompt


def parse_citations(llm_output: str, retrieved_chunks: list[dict]) -> list[dict]:
    """
    Extract source file citations from the LLM's answer text.

    The LLM is prompted to write citations in this format:
        [FILE: path/to/file.py, Lines 12-34]

    This function:
        1. Uses regex to find all [FILE: ...] markers in the LLM output
        2. Deduplicates them (same file+lines only appears once)
        3. Falls back to the raw retrieved chunk metadata if the LLM
           didn't include any citations (in case it forgets)

    Args:
        llm_output:       The full text response from the LLM.
        retrieved_chunks: The chunks that were passed to the LLM as context.

    Returns:
        List of citation dicts:
        [
            {"file": "src/auth.py", "lines": "42-58", "score": 0.87},
            ...
        ]
    """
    citations = []
    seen      = set()

    # Pattern: [FILE: path/to/file.py, Lines 10-20]
    # Also handles: [FILE: path, Lines 10] (single line)
    pattern = re.compile(
        r'\[FILE:\s*([^\],]+),\s*Lines?\s*(\d+(?:\s*[-–]\s*\d+)?)\]',
        re.IGNORECASE
    )

    for match in pattern.finditer(llm_output):
        file_path  = match.group(1).strip()
        line_range = match.group(2).strip().replace("–", "-").replace(" ", "")
        key        = f"{file_path}:{line_range}"

        if key in seen:
            continue
        seen.add(key)

        # Try to find the score for this citation from the retrieved chunks
        score = 0.0
        for chunk in retrieved_chunks:
            if chunk.get("file_path", "") == file_path:
                score = chunk.get("score", 0.0)
                break

        citations.append({
            "file":  file_path,
            "lines": line_range,
            "score": round(score, 4),
        })

    # Fallback: if LLM didn't cite anything, use the retrieved chunks directly
    if not citations and retrieved_chunks:
        logger.debug("LLM produced no citations — falling back to retrieved chunks")
        for chunk in retrieved_chunks:
            file_path  = chunk.get("file_path", "unknown")
            line_range = f"{chunk.get('start_line',0)}-{chunk.get('end_line',0)}"
            key        = f"{file_path}:{line_range}"
            if key not in seen:
                seen.add(key)
                citations.append({
                    "file":  file_path,
                    "lines": line_range,
                    "score": round(chunk.get("score", 0.0), 4),
                })

    return citations


def answer_question(
    question:  str,
    repo_name: str,
    top_k:     Optional[int]   = None,
    threshold: Optional[float] = None,
) -> dict:
    """
    The main RAG pipeline entry point.

    Takes a natural language question, retrieves relevant code chunks,
    and returns a cited AI-generated answer.

    Flow:
        1. Embed the question
        2. Search ChromaDB for top-k similar chunks
        3. Filter chunks below the similarity threshold
        4. Build a structured prompt
        5. Send to Groq Llama 3
        6. Parse citations from the response
        7. Return answer + sources

    Args:
        question:  The user's question, e.g. "how does login work?"
        repo_name: Collection name in ChromaDB, e.g. "psf__requests"
        top_k:     Number of chunks to retrieve. Defaults to settings.top_k_results.
        threshold: Min similarity score. Defaults to settings.similarity_threshold.

    Returns:
        dict with keys:
            "answer"   — string: the LLM's answer (with inline citations)
            "sources"  — list of {"file", "lines", "score"} dicts
            "chunks_retrieved" — int: how many chunks were found above threshold
            "repo_name"        — string: the repository that was queried

    Example return value:
        {
            "answer":  "The login function in src/auth.py (Lines 42-58) handles...",
            "sources": [{"file": "src/auth.py", "lines": "42-58", "score": 0.87}],
            "chunks_retrieved": 3,
            "repo_name": "psf__requests"
        }
    """
    from config.settings import settings, require_groq_key, require_chroma_index
    from core.embedder    import embed_text
    from core.vector_store import similarity_search
    from core.llm_client  import generate

    if top_k is None:
        top_k = settings.top_k_results
    if threshold is None:
        threshold = settings.similarity_threshold

    # ── Validate prerequisites ─────────────────────────────────────────────────
    require_groq_key()
    require_chroma_index(repo_name)

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    logger.info(f"RAG query: '{question[:80]}' | repo={repo_name} | top_k={top_k}")

    # ── Step 1: Embed the question ─────────────────────────────────────────────
    logger.debug("Embedding query...")
    query_vector = embed_text(question)

    # ── Step 2: Retrieve top-k chunks from ChromaDB ───────────────────────────
    logger.debug(f"Searching ChromaDB (top_k={top_k})...")
    raw_results = similarity_search(
        query_vector = query_vector,
        repo_name    = repo_name,
        top_k        = top_k,
    )

    # ── Step 3: Filter by similarity threshold ────────────────────────────────
    filtered = [r for r in raw_results if r.get("score", 0) >= threshold]

    logger.info(
        f"Retrieved {len(raw_results)} chunks, "
        f"{len(filtered)} passed threshold ({threshold})"
    )

    # ── Step 4: No relevant context found — return early ──────────────────────
    if not filtered:
        logger.info("No chunks above threshold — returning no-context answer")
        return {
            "answer":           NO_CONTEXT_ANSWER,
            "sources":          [],
            "chunks_retrieved": 0,
            "repo_name":        repo_name,
        }

    # ── Step 5: Build the prompt ───────────────────────────────────────────────
    filtered = _prioritize_chunks(filtered)
    context_chunks = _prepare_chunks_for_llm(filtered)
    prompt = build_prompt(question, context_chunks)
    logger.info(
        f"Prompt size: {len(prompt)} chars (~{len(prompt) // 4} tokens), "
        f"{len(context_chunks)} chunks sent to LLM"
    )

    # ── Step 6: Call the LLM ──────────────────────────────────────────────────
    logger.debug("Calling Groq LLM...")
    llm_answer = generate(prompt)

    # ── Step 7: Parse citations ───────────────────────────────────────────────
    sources = parse_citations(llm_answer, context_chunks)

    logger.info(f"Answer generated: {len(llm_answer)} chars, {len(sources)} citations")

    return {
        "answer":           llm_answer,
        "sources":          sources,
        "chunks_retrieved": len(filtered),
        "repo_name":        repo_name,
    }
