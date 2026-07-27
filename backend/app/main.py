"""
FastAPI application exposing two endpoints:

  POST /upload  - accepts a PDF/TXT file, chunks it, embeds it, stores it
  POST /query   - accepts a question, retrieves relevant chunks, streams
                  back an LLM-generated answer (Server-Sent Events)

Run with: uvicorn app.main:app --reload --port 8000
"""
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.loaders import load_document
from app.chunking import chunk_text
from app.embeddings import embed_texts, embed_query
from app.vectorstore import add_chunks, query as vector_query, collection_stats, clear_collection
from app.llm_client import stream_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="RAG Doc Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", **collection_stats()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Persist to a temp file since our loaders work off file paths
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        raw_text = load_document(tmp_path)
        if not raw_text.strip():
            raise HTTPException(400, "No extractable text found in file")

        chunks = chunk_text(raw_text, source=file.filename)
        embeddings = embed_texts([c.text for c in chunks])
        stored = add_chunks(chunks, embeddings)

        return {"filename": file.filename, "chunks_stored": stored}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/query")
def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    query_embedding = embed_query(req.question)
    matches = vector_query(query_embedding)

    if not matches:
        def empty_stream():
            yield "I don't have any documents indexed yet -- upload one first."
        return StreamingResponse(empty_stream(), media_type="text/plain")

    def token_stream():
        try:
            for token in stream_answer(req.question, matches):
                yield token
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming answer failed")
            yield f"\n\n[Error generating answer: {exc}]"

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.delete("/documents")
def reset_documents():
    clear_collection()
    return {"status": "cleared"}
