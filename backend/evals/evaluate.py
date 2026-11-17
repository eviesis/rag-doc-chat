"""
Retrieval evaluation harness.

Measures how well the retrieval half of the RAG pipeline finds the right
chunk, and how long it takes, for each combination of:

    vector store : chroma | faiss
    reranking    : off | on (Hugging Face cross-encoder)

It calls app.retrieval.retrieve(), the exact function the API uses, so the
numbers describe the real system. No LLM key is needed.

Ground truth
------------
Each question has an `answer_contains` string. A retrieved chunk counts as
relevant if it contains that string (case-insensitive, whitespace-normalised).
This is a simple, transparent proxy for relevance -- good enough to compare
configurations, but it is not a full answer-quality evaluation (that would
need the LLM and a judge). Write your own questions for your own documents.

Metrics
-------
  Hit@1   fraction of questions whose #1 chunk is relevant
  Hit@k   fraction where any of the top-k chunks is relevant
  MRR     mean of 1/rank of the first relevant chunk (0 if none)
  latency per-query retrieval time (embed + search + optional rerank)

Usage (run from the backend/ folder):

    # compare all four configurations side by side
    python -m evals.evaluate --compare \
        --doc ../sample_docs/engineering_handbook.txt \
        --questions ../sample_docs/engineering_handbook_questions.json \
        --chunk-size 100 --overlap 20 --top-k 3

    # a single configuration
    python -m evals.evaluate --store faiss --rerank --doc ... --questions ...
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def run_single(args) -> dict:
    """Ingest the document and evaluate one (store, rerank) configuration."""
    workdir = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    # Configure BEFORE importing app modules: settings are read at import time.
    os.environ["VECTOR_STORE"] = args.store
    os.environ["CHROMA_PERSIST_DIR"] = str(workdir / "chroma")
    os.environ["FAISS_INDEX_DIR"] = str(workdir / "faiss")
    os.environ["RERANK_ENABLED"] = "true" if args.rerank else "false"
    sys.path.insert(0, str(BACKEND_DIR))

    from app.loaders import load_document
    from app.chunking import chunk_text
    from app.embeddings import embed_texts
    from app.vectorstore import add_chunks
    from app.retrieval import retrieve

    doc_path = Path(args.doc)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    # --- ingestion (timed) ---
    t0 = time.perf_counter()
    chunks = chunk_text(
        load_document(doc_path),
        source=doc_path.name,
        chunk_size_tokens=args.chunk_size,
        overlap_tokens=args.overlap,
    )
    add_chunks(chunks, embed_texts([c.text for c in chunks]))
    ingest_s = time.perf_counter() - t0

    # Warm-up so model loading is not counted in per-query latency.
    retrieve(questions[0]["question"], top_k=args.top_k, rerank_enabled=args.rerank)

    hits_at_1 = hits_at_k = 0
    reciprocal_ranks, latencies_ms, misses = [], [], []

    for q in questions:
        t = time.perf_counter()
        matches = retrieve(q["question"], top_k=args.top_k, rerank_enabled=args.rerank)
        latencies_ms.append((time.perf_counter() - t) * 1000)

        needle = _norm(q["answer_contains"])
        rank = next((i for i, m in enumerate(matches) if needle in _norm(m["text"])), None)

        if rank is None:
            reciprocal_ranks.append(0.0)
            misses.append(q["question"])
        else:
            reciprocal_ranks.append(1.0 / (rank + 1))
            hits_at_k += 1
            if rank == 0:
                hits_at_1 += 1

    n = len(questions)
    latencies_ms.sort()
    return {
        "store": args.store,
        "rerank": args.rerank,
        "questions": n,
        "chunks": len(chunks),
        "top_k": args.top_k,
        "hit_at_1": hits_at_1 / n,
        "hit_at_k": hits_at_k / n,
        "mrr": sum(reciprocal_ranks) / n,
        "latency_ms_median": latencies_ms[n // 2],
        "latency_ms_p95": latencies_ms[min(n - 1, int(0.95 * n))],
        "ingest_seconds": ingest_s,
        "misses": misses,
    }


def _passthrough(args) -> list:
    return [
        "--doc", args.doc, "--questions", args.questions,
        "--chunk-size", str(args.chunk_size), "--overlap", str(args.overlap),
        "--top-k", str(args.top_k),
    ]


def run_compare(args) -> list:
    results = []
    for store in ("chroma", "faiss"):
        for rerank in (False, True):
            out = Path(tempfile.mkstemp(suffix=".json")[1])
            cmd = [sys.executable, "-m", "evals.evaluate", "--store", store,
                   "--json-out", str(out)] + _passthrough(args)
            if rerank:
                cmd.append("--rerank")
            print(f"running: store={store} rerank={'on' if rerank else 'off'} ...", flush=True)
            proc = subprocess.run(cmd, cwd=BACKEND_DIR)
            if proc.returncode != 0:
                print(f"  FAILED (exit {proc.returncode}) -- see error above", file=sys.stderr)
                continue
            results.append(json.loads(out.read_text()))
            out.unlink(missing_ok=True)
    return results


def print_table(results: list) -> None:
    if not results:
        print("No results.")
        return
    r0 = results[0]
    print(f"\n{r0['questions']} questions, {r0['chunks']} chunks, top_k={r0['top_k']}\n")
    header = f"{'store':<8}{'rerank':<8}{'Hit@1':>8}{'Hit@k':>8}{'MRR':>8}{'p50 ms':>10}{'p95 ms':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['store']:<8}{'on' if r['rerank'] else 'off':<8}"
            f"{r['hit_at_1']:>8.2f}{r['hit_at_k']:>8.2f}{r['mrr']:>8.2f}"
            f"{r['latency_ms_median']:>10.1f}{r['latency_ms_p95']:>10.1f}"
        )
    for r in results:
        if r["misses"]:
            tag = f"{r['store']}/{'rerank' if r['rerank'] else 'no-rerank'}"
            print(f"\nMissed by {tag}:")
            for m in r["misses"]:
                print(f"  - {m}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doc", required=True, help="path to a .txt/.md/.pdf document")
    p.add_argument("--questions", required=True, help="JSON list of {question, answer_contains}")
    p.add_argument("--store", choices=["chroma", "faiss"], default="chroma")
    p.add_argument("--rerank", action="store_true", help="enable cross-encoder reranking")
    p.add_argument("--compare", action="store_true", help="run all four store/rerank combinations")
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--json-out", help="write the single-run result to this JSON file")
    args = p.parse_args()

    if args.compare:
        print_table(run_compare(args))
        return

    result = run_single(args)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result), encoding="utf-8")
    else:
        print_table([result])


if __name__ == "__main__":
    main()
