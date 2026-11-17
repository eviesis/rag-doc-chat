"""
Second-stage reranking with a Hugging Face cross-encoder.

Why this exists
---------------
The first stage (vector search) uses a *bi-encoder*: the question and every
chunk are embedded independently and compared by cosine similarity. That is
fast, but the model never sees the question and chunk together.

A *cross-encoder* reads (question, chunk) as one input and outputs a relevance
score. It is much more accurate but too slow to run over the whole corpus, so
the standard pattern is:

    vector search -> top ~12 candidates -> cross-encoder rerank -> top ~4

The model is loaded lazily so the app starts fast when reranking is disabled.
"""
from functools import lru_cache
from typing import List, Dict, Any

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.RERANK_MODEL_NAME)


def rerank(question: str, matches: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
    """Re-score `matches` against `question` and return the best `top_k`, best first."""
    top_k = top_k or settings.TOP_K_CHUNKS
    if not matches:
        return []

    model = _get_model()
    scores = model.predict([(question, m["text"]) for m in matches])

    scored = []
    for m, s in zip(matches, scores):
        scored.append({**m, "rerank_score": float(s)})
    scored.sort(key=lambda m: m["rerank_score"], reverse=True)
    return scored[:top_k]
