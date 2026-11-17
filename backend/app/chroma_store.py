"""
Thin wrapper around a persistent ChromaDB collection.

Isolating this in one module means the rest of the app never talks to
Chroma's API directly -- if you swap in Pinecone/Weaviate/pgvector later,
this is the only file that needs to change.
"""
from typing import List, Dict, Any

import chromadb

from app.config import settings
from app.chunking import Chunk

_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
_collection = _client.get_or_create_collection(name=settings.CHROMA_COLLECTION_NAME)


def add_chunks(chunks: List[Chunk], embeddings: List[List[float]]) -> int:
    if not chunks:
        return 0

    ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)


def query(embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.TOP_K_CHUNKS
    results = _collection.query(query_embeddings=[embedding], n_results=top_k)

    matches = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        matches.append({"text": doc, "metadata": meta, "distance": dist})
    return matches


def collection_stats() -> Dict[str, Any]:
    return {"count": _collection.count(), "name": settings.CHROMA_COLLECTION_NAME}


def clear_collection() -> None:
    _client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    global _collection
    _collection = _client.get_or_create_collection(name=settings.CHROMA_COLLECTION_NAME)
