"""
FAISS-backed vector store.

FAISS (Facebook AI Similarity Search) is a *similarity search library*, not
a database: it stores vectors and answers nearest-neighbour queries, but it
has no notion of text or metadata. So this module pairs the FAISS index with
a small JSON "docstore" that maps each vector's integer id back to its chunk
text and metadata.

Design notes (worth being able to explain in an interview):
  - Embeddings are L2-normalised, so inner product == cosine similarity.
    We therefore use IndexFlatIP (exact search) rather than an L2 index.
  - IndexFlatIP is exact (brute force). That's the right choice at this scale
    (thousands of chunks). For millions of vectors you would switch to an
    approximate index such as IndexIVFFlat or IndexHNSWFlat, trading a little
    recall for much lower latency.
  - IndexIDMap lets us assign our own integer ids, which makes upsert
    (re-uploading the same file) possible: remove old ids, add the new ones.
  - The index and docstore are persisted to disk after every write.
"""
import json
import threading
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np

from app.config import settings
from app.chunking import Chunk

_INDEX_FILE = "index.faiss"
_DOCS_FILE = "docstore.json"

_lock = threading.RLock()
_dir = Path(settings.FAISS_INDEX_DIR)
_dir.mkdir(parents=True, exist_ok=True)

_index = None            # faiss.IndexIDMap
_records: Dict[int, Dict[str, Any]] = {}   # int id -> {"key", "text", "metadata"}
_key_to_id: Dict[str, int] = {}            # "source::chunk_index" -> int id
_next_id: int = 0


def _load() -> None:
    global _index, _records, _key_to_id, _next_id
    index_path = _dir / _INDEX_FILE
    docs_path = _dir / _DOCS_FILE
    if index_path.exists() and docs_path.exists():
        _index = faiss.read_index(str(index_path))
        raw = json.loads(docs_path.read_text(encoding="utf-8"))
        _records = {int(k): v for k, v in raw["records"].items()}
        _next_id = raw["next_id"]
        _key_to_id = {rec["key"]: i for i, rec in _records.items()}


def _save() -> None:
    if _index is None:
        return
    faiss.write_index(_index, str(_dir / _INDEX_FILE))
    payload = {"next_id": _next_id, "records": {str(k): v for k, v in _records.items()}}
    (_dir / _DOCS_FILE).write_text(json.dumps(payload), encoding="utf-8")


def _ensure_index(dim: int) -> None:
    global _index
    if _index is None:
        _index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    elif _index.d != dim:
        raise ValueError(
            f"Embedding dimension {dim} does not match the stored index "
            f"dimension {_index.d}. You changed EMBEDDING_MODEL_NAME -- clear "
            f"the index (DELETE /documents) and re-upload your documents."
        )


def add_chunks(chunks: List[Chunk], embeddings: List[List[float]]) -> int:
    global _next_id
    if not chunks:
        return 0

    vectors = np.asarray(embeddings, dtype="float32")
    with _lock:
        _ensure_index(vectors.shape[1])

        # Upsert semantics: drop any existing vectors for the same chunk keys.
        keys = [f"{c.source}::{c.chunk_index}" for c in chunks]
        stale = [_key_to_id[k] for k in keys if k in _key_to_id]
        if stale:
            _index.remove_ids(np.asarray(stale, dtype="int64"))
            for i in stale:
                _key_to_id.pop(_records.pop(i)["key"], None)

        ids = np.arange(_next_id, _next_id + len(chunks), dtype="int64")
        _next_id += len(chunks)
        _index.add_with_ids(vectors, ids)

        for i, c, k in zip(ids.tolist(), chunks, keys):
            _records[i] = {
                "key": k,
                "text": c.text,
                "metadata": {"source": c.source, "chunk_index": c.chunk_index},
            }
            _key_to_id[k] = i

        _save()
    return len(chunks)


def query(embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.TOP_K_CHUNKS
    with _lock:
        if _index is None or _index.ntotal == 0:
            return []
        q = np.asarray([embedding], dtype="float32")
        scores, ids = _index.search(q, min(top_k, _index.ntotal))

        matches = []
        for score, i in zip(scores[0].tolist(), ids[0].tolist()):
            if i == -1:
                continue
            rec = _records[i]
            # Report as a distance (lower = closer) so it matches the Chroma backend.
            matches.append(
                {"text": rec["text"], "metadata": rec["metadata"], "distance": 1.0 - score}
            )
        return matches


def collection_stats() -> Dict[str, Any]:
    with _lock:
        return {"count": 0 if _index is None else int(_index.ntotal), "name": "faiss"}


def clear_collection() -> None:
    global _index, _records, _key_to_id, _next_id
    with _lock:
        _index = None
        _records = {}
        _key_to_id = {}
        _next_id = 0
        for f in (_INDEX_FILE, _DOCS_FILE):
            (_dir / f).unlink(missing_ok=True)


_load()
