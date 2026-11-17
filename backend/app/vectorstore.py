"""
Vector store facade.

The rest of the app imports from here and never needs to know which backend
is active. Pick one with VECTOR_STORE=chroma|faiss in .env.

Backends are imported lazily so that, for example, running with FAISS does not
create a ChromaDB directory (and vice versa).
"""
from app.config import settings

if settings.VECTOR_STORE == "faiss":
    from app.faiss_store import add_chunks, query, collection_stats, clear_collection
elif settings.VECTOR_STORE == "chroma":
    from app.chroma_store import add_chunks, query, collection_stats, clear_collection
else:
    raise ValueError(
        f"Unknown VECTOR_STORE '{settings.VECTOR_STORE}' -- use 'chroma' or 'faiss'"
    )

__all__ = ["add_chunks", "query", "collection_stats", "clear_collection"]
