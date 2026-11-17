"""
Unit tests for the FAISS vector store. They use hand-made vectors, so no
embedding model is downloaded.

Run from the backend/ folder:   pip install pytest   then   pytest -q
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIM = 8


def _chunks(source, texts):
    from app.chunking import Chunk
    return [Chunk(text=t, chunk_index=i, source=source, token_count=len(t.split()))
            for i, t in enumerate(texts)]


def _vec(i):
    """A unit vector pointing along axis i (orthogonal to every other axis)."""
    v = np.zeros(DIM, dtype="float32")
    v[i] = 1.0
    return v.tolist()


@pytest.fixture
def fs(tmp_path, monkeypatch):
    """A freshly loaded faiss_store pointed at an empty temp directory."""
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path))
    import app.config
    importlib.reload(app.config)
    import app.faiss_store
    return importlib.reload(app.faiss_store)


def test_returns_nearest_neighbour_first(fs):
    chunks = _chunks("doc.txt", ["alpha", "beta", "gamma"])
    fs.add_chunks(chunks, [_vec(0), _vec(1), _vec(2)])

    hits = fs.query(_vec(1), top_k=3)

    assert hits[0]["text"] == "beta"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)   # identical vector
    assert hits[1]["distance"] == pytest.approx(1.0, abs=1e-6)   # orthogonal vector


def test_reuploading_same_source_replaces_instead_of_duplicating(fs):
    fs.add_chunks(_chunks("doc.txt", ["old text"]), [_vec(0)])
    fs.add_chunks(_chunks("doc.txt", ["new text"]), [_vec(0)])

    assert fs.collection_stats()["count"] == 1
    assert fs.query(_vec(0), top_k=1)[0]["text"] == "new text"


def test_different_sources_do_not_collide(fs):
    fs.add_chunks(_chunks("a.txt", ["from a"]), [_vec(0)])
    fs.add_chunks(_chunks("b.txt", ["from b"]), [_vec(1)])
    assert fs.collection_stats()["count"] == 2


def test_index_survives_a_restart(fs):
    fs.add_chunks(_chunks("doc.txt", ["alpha", "beta"]), [_vec(0), _vec(1)])

    reloaded = importlib.reload(sys.modules["app.faiss_store"])   # simulates a new process

    assert reloaded.collection_stats()["count"] == 2
    assert reloaded.query(_vec(0), top_k=1)[0]["text"] == "alpha"


def test_metadata_is_returned(fs):
    fs.add_chunks(_chunks("doc.txt", ["alpha"]), [_vec(0)])
    meta = fs.query(_vec(0), top_k=1)[0]["metadata"]
    assert meta == {"source": "doc.txt", "chunk_index": 0}


def test_mismatched_embedding_dimension_gives_a_clear_error(fs):
    fs.add_chunks(_chunks("doc.txt", ["alpha"]), [_vec(0)])
    with pytest.raises(ValueError, match="EMBEDDING_MODEL_NAME"):
        fs.add_chunks(_chunks("other.txt", ["beta"]), [[1.0, 0.0, 0.0]])


def test_query_on_empty_index_returns_nothing(fs):
    assert fs.query(_vec(0)) == []


def test_clear_collection_empties_the_store(fs):
    fs.add_chunks(_chunks("doc.txt", ["alpha"]), [_vec(0)])
    fs.clear_collection()
    assert fs.collection_stats()["count"] == 0
    assert fs.query(_vec(0)) == []
