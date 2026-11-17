"""
Single entry point for retrieval, shared by the API and the eval script so
that what gets measured is exactly what runs in production.

    embed query -> vector search (candidates) -> optional cross-encoder rerank
"""
from typing import List, Dict, Any

from app.config import settings
from app.embeddings import embed_query
from app.vectorstore import query as vector_query


def retrieve(question: str, top_k: int = None, rerank_enabled: bool = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.TOP_K_CHUNKS
    if rerank_enabled is None:
        rerank_enabled = settings.RERANK_ENABLED

    query_embedding = embed_query(question)

    if not rerank_enabled:
        return vector_query(query_embedding, top_k=top_k)

    # Over-fetch candidates, then let the cross-encoder pick the best top_k.
    from app.reranker import rerank
    candidates = vector_query(query_embedding, top_k=max(settings.RETRIEVE_CANDIDATES, top_k))
    return rerank(question, candidates, top_k=top_k)
