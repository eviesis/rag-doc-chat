"""
Embedding generation using a local, open-source sentence-transformers
model (all-MiniLM-L6-v2 by default).

Running embeddings locally means no API key or per-call cost for the
retrieval half of the pipeline -- only the final generation step calls
out to Claude. The model is loaded once and reused (module-level cache)
since loading it is the expensive part.
"""
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts. Returns one vector (list of floats) per input text."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
