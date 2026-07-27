"""
Text chunking utilities.

Splits raw document text into overlapping chunks sized by *token* count
(using tiktoken for an approximate, model-agnostic tokenizer) rather than
raw character count, since token count is what actually matters for
embedding models and LLM context windows.
"""
from dataclasses import dataclass
from typing import List

import tiktoken

from app.config import settings

# cl100k_base is a reasonable general-purpose tokenizer for chunk sizing;
# it doesn't need to exactly match the embedding model's own tokenizer.
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source: str
    token_count: int


def _encode(text: str) -> List[int]:
    return _ENCODING.encode(text)


def _decode(tokens: List[int]) -> str:
    return _ENCODING.decode(tokens)


def chunk_text(
    text: str,
    source: str,
    chunk_size_tokens: int = None,
    overlap_tokens: int = None,
) -> List[Chunk]:
    """
    Split `text` into overlapping chunks.

    Overlap exists so that context isn't lost at chunk boundaries -- a
    sentence that would otherwise be split awkwardly between two chunks
    still appears in full in at least one of them.
    """
    chunk_size_tokens = chunk_size_tokens or settings.CHUNK_SIZE_TOKENS
    overlap_tokens = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS

    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    tokens = _encode(text)
    chunks: List[Chunk] = []

    start = 0
    index = 0
    step = chunk_size_tokens - overlap_tokens

    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = _decode(chunk_tokens).strip()

        if chunk_str:
            chunks.append(
                Chunk(
                    text=chunk_str,
                    chunk_index=index,
                    source=source,
                    token_count=len(chunk_tokens),
                )
            )
            index += 1

        if end == len(tokens):
            break
        start += step

    return chunks
