"""
LLM client wrapper supporting two interchangeable providers:

  - "anthropic" : Claude API (paid after a small trial credit)
  - "groq"      : Groq API (generous free tier, open-source models
                  like Llama 3.3, OpenAI-compatible SDK)

Switch providers by setting LLM_PROVIDER=anthropic|groq in .env --
no other code needs to change. Both paths share the same retry/backoff,
rate-limiting, and streaming behavior so the rest of the app (main.py)
never needs to know which provider is active.
"""
import time
import logging
from typing import List, Dict, Generator

from app.config import settings

logger = logging.getLogger("llm_client")

# --- Rate limiting (shared across both providers) --------------------------

_last_call_time: float = 0.0


def _respect_rate_limit() -> None:
    global _last_call_time
    elapsed = time.monotonic() - _last_call_time
    wait = settings.LLM_MIN_SECONDS_BETWEEN_CALLS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.monotonic()


def build_rag_prompt_text(question: str, retrieved_chunks: List[Dict]) -> str:
    """Builds the combined system+context+question text used by both providers."""
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['text']}"
        for c in retrieved_chunks
    )
    return (
        "You are a helpful assistant answering questions using ONLY the "
        "provided context. If the answer isn't in the context, say you "
        "don't know rather than guessing.\n\n"
        f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"
    )


# --- Anthropic provider ------------------------------------------------------

def _stream_anthropic(prompt_text: str) -> Generator[str, None, None]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
            return True
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
            return True
        return False

    attempt = 0
    while True:
        _respect_rate_limit()
        try:
            with client.messages.stream(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=settings.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt_text}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
            return
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if is_retryable(exc) and attempt <= settings.LLM_MAX_RETRIES:
                backoff = settings.LLM_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Anthropic call failed (attempt %s/%s): %s. Retrying in %.1fs",
                    attempt, settings.LLM_MAX_RETRIES, exc, backoff,
                )
                time.sleep(backoff)
                continue
            logger.error("Anthropic call failed permanently after %s attempts: %s", attempt, exc)
            raise


# --- Groq provider ------------------------------------------------------------

def _stream_groq(prompt_text: str) -> Generator[str, None, None]:
    from groq import Groq
    import groq as groq_module

    client = Groq(api_key=settings.GROQ_API_KEY)

    def is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (groq_module.RateLimitError, groq_module.APIConnectionError)):
            return True
        if isinstance(exc, groq_module.APIStatusError) and exc.status_code >= 500:
            return True
        return False

    attempt = 0
    while True:
        _respect_rate_limit()
        try:
            stream = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                max_tokens=settings.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt_text}],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if is_retryable(exc) and attempt <= settings.LLM_MAX_RETRIES:
                backoff = settings.LLM_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Groq call failed (attempt %s/%s): %s. Retrying in %.1fs",
                    attempt, settings.LLM_MAX_RETRIES, exc, backoff,
                )
                time.sleep(backoff)
                continue
            logger.error("Groq call failed permanently after %s attempts: %s", attempt, exc)
            raise


# --- Public interface ---------------------------------------------------------

def stream_answer(question: str, retrieved_chunks: List[Dict]) -> Generator[str, None, None]:
    """
    Yields answer text incrementally as it streams back from whichever
    provider is configured (LLM_PROVIDER=anthropic|groq in .env).
    Retries the *connection attempt* with exponential backoff; once
    streaming has started, a mid-stream failure surfaces to the caller
    rather than silently retrying, to avoid duplicating partial output.
    """
    prompt_text = build_rag_prompt_text(question, retrieved_chunks)

    if settings.LLM_PROVIDER == "groq":
        yield from _stream_groq(prompt_text)
    elif settings.LLM_PROVIDER == "anthropic":
        yield from _stream_anthropic(prompt_text)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}' -- use 'anthropic' or 'groq'"
        )
