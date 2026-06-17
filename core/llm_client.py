"""
core/llm_client.py  [NEW — Week 4]
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Wraps the Groq API to generate text completions using Llama 3.

WHY GROQ?
    Groq provides Llama 3 inference at extremely high speed (hundreds of
    tokens per second) on their custom hardware. They offer a generous
    free tier — enough for all development and demo use.

    Sign up at: console.groq.com (free, no credit card required)

MODEL USED:
    llama3-8b-8192  (default)
    - Context window: 8,192 tokens
    - Speed: very fast
    - Quality: excellent for code explanation tasks
    - Cost: free on Groq's current tier

FUNCTIONS:
    generate(prompt)        — send a prompt, get a text response
    generate_with_messages(messages) — send a full message list (for multi-turn)

RETRY BEHAVIOR:
    On rate limit (429) or server error (500+):
        - Attempt 1: immediate
        - Attempt 2: wait 2 seconds
        - Attempt 3: wait 4 seconds
    After 3 failed attempts: raises the original exception
─────────────────────────────────────────────────────────────────────────────
"""

import time
import logging

logger = logging.getLogger(__name__)

# ─── Lazy singleton Groq client ───────────────────────────────────────────────
_groq_client = None


def get_groq_client():
    """
    Return a lazy-loaded Groq client singleton.

    The client is created once on first call and reused.

    Raises:
        ImportError: If the groq library is not installed.
        RuntimeError: If GROQ_API_KEY is missing.
    """
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "groq library not installed.\n"
            "Fix: pip install groq  (or pip install -r requirements.txt)"
        )

    from config.settings import settings, require_groq_key
    require_groq_key()

    _groq_client = Groq(api_key=settings.groq_api_key)
    logger.info(f"Groq client initialized with model: {settings.llm_model}")
    return _groq_client


def generate(
    prompt:     str,
    model:      str  = None,
    temperature: float = None,
    max_tokens: int  = None,
) -> str:
    """
    Send a prompt to Groq and return the generated text.

    This is the main function called by rag_engine.py.

    The prompt contains:
        - System instructions (how the LLM should behave)
        - Retrieved code chunks (the context)
        - The user's question
        - Instructions to cite sources

    Args:
        prompt:      The full prompt string to send to the LLM.
        model:       Groq model identifier. Defaults to settings.llm_model.
        temperature: Sampling temperature. Defaults to settings.llm_temperature.
        max_tokens:  Max tokens to generate. Defaults to settings.llm_max_tokens.

    Returns:
        The LLM's response as a plain text string.

    Raises:
        RuntimeError: After 3 failed retry attempts.

    Example:
        response = generate("Explain what this function does: def login(): ...")
        print(response)
        # "The login() function handles user authentication by..."
    """
    from config.settings import settings

    model       = model       or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature
    max_tokens  = max_tokens  or settings.llm_max_tokens

    client     = get_groq_client()
    last_error = None

    for attempt in range(1, 4):   # up to 3 attempts
        try:
            logger.debug(f"Groq request attempt {attempt}/3 (model={model})")
            start    = time.time()
            response = client.chat.completions.create(
                model    = model,
                messages = [{"role": "user", "content": prompt}],
                temperature = temperature,
                max_tokens  = max_tokens,
            )
            elapsed  = time.time() - start
            content  = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else "?"
            logger.info(
                f"Groq response: {len(content)} chars, "
                f"~{tokens_used} tokens, {elapsed:.2f}s"
            )
            return content

        except Exception as e:
            last_error  = e
            error_str   = str(e).lower()

            # Rate limit, payload too large, or server error — retry only when useful
            if (
                "rate_limit" in error_str
                or "429" in error_str
                or "500" in error_str
            ) and "413" not in error_str and "too large" not in error_str:
                wait = 2 ** (attempt - 1)   # 1s, 2s, 4s
                logger.warning(
                    f"Groq error on attempt {attempt}: {e}. "
                    f"Retrying in {wait}s..."
                )
                if attempt < 3:
                    time.sleep(wait)
                continue

            # Non-retryable error (bad API key, invalid model, payload too large, etc.)
            logger.error(f"Non-retryable Groq error: {e}")
            if "413" in error_str or "too large" in error_str:
                raise RuntimeError(
                    "Groq request too large for the free tier token limit.\n"
                    "The retrieved code context was trimmed automatically — "
                    "try a more specific question or re-index the repository."
                ) from e
            raise RuntimeError(
                f"Groq API error: {e}\n"
                "Check: Is GROQ_API_KEY correct in .env? Is the model name valid?"
            ) from e

    raise RuntimeError(
        f"Groq API failed after 3 attempts. Last error: {last_error}\n"
        "Check your internet connection and Groq API status at status.groq.com"
    ) from last_error


def generate_with_messages(
    messages:    list[dict],
    model:       str   = None,
    temperature: float = None,
    max_tokens:  int   = None,
) -> str:
    """
    Send a full message list to Groq (for multi-turn conversation use).

    Args:
        messages:    List of {"role": "user"/"assistant"/"system", "content": "..."} dicts.
        model:       Groq model identifier.
        temperature: Sampling temperature.
        max_tokens:  Max tokens to generate.

    Returns:
        LLM response string.

    Example:
        messages = [
            {"role": "system", "content": "You are a helpful code assistant."},
            {"role": "user",   "content": "What does this function do?"},
        ]
        response = generate_with_messages(messages)
    """
    from config.settings import settings

    model       = model       or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature
    max_tokens  = max_tokens  or settings.llm_max_tokens

    client     = get_groq_client()
    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                temperature = temperature,
                max_tokens  = max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            error_str  = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str or "500" in error_str:
                wait = 2 ** (attempt - 1)
                if attempt < 3:
                    time.sleep(wait)
                continue
            raise RuntimeError(f"Groq API error: {e}") from e

    raise RuntimeError(
        f"Groq API failed after 3 attempts: {last_error}"
    ) from last_error
