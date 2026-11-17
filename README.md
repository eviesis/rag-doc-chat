# RAG Doc Chat

A small, complete Retrieval-Augmented Generation (RAG) application: upload
a document, ask questions about it, get answers streamed back from Claude,
grounded only in the content you uploaded.

Built to demonstrate a specific set of skills end-to-end:
**chunking → embeddings → vector retrieval → LLM integration (streaming,
retries, rate limiting) → a React chat UI.**

## Architecture

```
┌─────────────┐      ┌──────────────────────────────────────────┐
│   React     │      │              FastAPI backend              │
│  Chat UI    │◄────►│                                            │
│             │ HTTP │  /upload                                   │
│  - upload   │      │    file → text (loaders.py)                │
│  - chat     │      │         → chunks (chunking.py, tiktoken)   │
│             │      │         → embeddings (embeddings.py,       │
│             │      │           sentence-transformers, local)    │
│             │      │         → stored (vectorstore.py, Chroma)  │
│             │      │                                            │
│             │      │  /query                                    │
│             │      │    question → embed → retrieve top-k       │
│             │      │             → prompt built with context     │
│             │      │             → Claude API (llm_client.py)   │
│             │      │               - streaming                  │
│             │      │               - retry w/ exponential       │
│             │      │                 backoff on 429/5xx          │
│             │      │               - client-side rate limiting  │
│             │      │             → tokens streamed back to UI   │
└─────────────┘      └──────────────────────────────────────────┘
```

## Why these choices

- **Embeddings run locally** (`sentence-transformers`, open-source,
  `all-MiniLM-L6-v2`) — no API key or per-call cost for retrieval, only
  the final answer generation calls out to Claude.
- **Chroma** as the vector store — zero-setup, persists to disk, good
  enough for a single-user/demo-scale app. Swappable for Pinecone/Weaviate
  by only touching `vectorstore.py`.
- **Streaming** — the backend streams tokens from Claude straight through
  to the frontend via a chunked HTTP response, so answers appear
  incrementally instead of after a long wait.
- **Retries + backoff** — transient failures (rate limits, 5xx, connection
  drops) are retried with exponential backoff before giving up; once
  streaming has actually started, a mid-stream failure is surfaced rather
  than silently retried (to avoid duplicating partial output).
- **Client-side rate limiting** — a minimum spacing between calls is
  enforced so the app doesn't hammer the API under bursty use.

## Setup

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env: set LLM_PROVIDER (anthropic or groq) and paste the matching API key
# Groq is free -- see "Getting an API key" below

uvicorn app.main:app --reload --port 8000
```

First run will download the embedding model (~90MB) — this only happens once.

### Getting an API key

**Groq (free, recommended for testing):**
1. Go to `console.groq.com`, sign up
2. Go to **API Keys**, click **Create API Key**
3. Paste it into `.env` as `GROQ_API_KEY`, set `LLM_PROVIDER=groq`

**Anthropic (small paid trial credit):**
1. Go to `console.anthropic.com`, sign up, verify phone number
2. Go to **Settings → API Keys**, generate a key
3. Paste it into `.env` as `ANTHROPIC_API_KEY`, set `LLM_PROVIDER=anthropic`

The rest of the app (chunking, embeddings, retrieval, retries, streaming) behaves
identically regardless of which provider you pick -- only `llm_client.py` branches
on `LLM_PROVIDER`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

### 3. Try it

1. Click **Upload document** and pick a PDF or `.txt` file (there's a
   sample at `sample_docs/sample_notes.txt` to try immediately).
2. Ask a question about its contents in the chat box.
3. Watch the answer stream in, grounded in the retrieved chunks.

## Retrieval options (FAISS, Hugging Face reranking) and evaluation

Everything below is switched with environment variables in `backend/.env`.

**Vector store: Chroma or FAISS** (`VECTOR_STORE=chroma|faiss`)
FAISS is a similarity-search *library*, not a database, so `faiss_store.py` pairs a
FAISS index (`IndexIDMap` over exact `IndexFlatIP`, cosine similarity on normalised
vectors) with a small JSON docstore for text and metadata. Both backends sit behind
`vectorstore.py`, so the rest of the app is unchanged. Exact search is the right choice
at this scale; at millions of vectors you would move to an approximate index such as
`IndexIVFFlat` or `IndexHNSWFlat`.

**Embeddings: Hugging Face models** (`EMBEDDING_MODEL_NAME`)
Default is `all-MiniLM-L6-v2`. `BAAI/bge-small-en-v1.5` is a stronger drop-in; set
`EMBEDDING_QUERY_PREFIX` as noted in `.env.example`. Changing the model changes the
vector size, so clear the index (`DELETE /documents`) and re-upload.

**Reranking: Hugging Face cross-encoder** (`RERANK_ENABLED=true`)
The vector search over-fetches `RETRIEVE_CANDIDATES` chunks, then
`cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (question, chunk) pair together and
keeps the best `TOP_K_CHUNKS`. This is slower than vector search alone but usually ranks
the right chunk higher.

**Evaluate it instead of guessing** (no LLM key needed):

```bash
cd backend
python -m evals.evaluate --compare \
  --doc ../sample_docs/engineering_handbook.txt \
  --questions ../sample_docs/engineering_handbook_questions.json \
  --chunk-size 100 --overlap 20 --top-k 3
```

This prints Hit@1, Hit@k, MRR and p50/p95 retrieval latency for Chroma/FAISS with
reranking on/off. The bundled handbook is a small fictional demo document; for
meaningful numbers, run it on a real multi-page document with your own questions
(`answer_contains` is the text a relevant chunk must contain). It measures *retrieval*
only, not final answer quality.

**Tests:** `pip install pytest`, then `pytest -q` from `backend/`.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST (multipart) | Upload a `.pdf`/`.txt`/`.md` file; chunks, embeds, and stores it |
| `/query` | POST (JSON `{"question": "..."}`) | Retrieves relevant chunks and streams back an answer |
| `/health` | GET | Basic health check + current stored chunk count |
| `/documents` | DELETE | Clears the vector store |

## Project structure

```
rag-doc-chat/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── config.py        # env-driven settings
│   │   ├── loaders.py       # PDF/text extraction
│   │   ├── chunking.py      # token-based overlapping chunker
│   │   ├── embeddings.py    # local Hugging Face sentence-transformers wrapper
│   │   ├── vectorstore.py   # picks the backend (chroma | faiss)
│   │   ├── chroma_store.py  # Chroma backend
│   │   ├── faiss_store.py   # FAISS backend (+ JSON docstore)
│   │   ├── reranker.py      # optional Hugging Face cross-encoder rerank
│   │   ├── retrieval.py     # embed -> search -> (rerank); shared with eval
│   │   └── llm_client.py    # LLM client (Anthropic/Groq): streaming + retry + rate limit
│   ├── evals/
│   │   └── evaluate.py      # Hit@k / MRR / latency across configurations
│   ├── tests/
│   │   └── test_faiss_store.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── FileUpload.jsx
│       │   └── MessageList.jsx
│       └── index.css
└── sample_docs/
    ├── sample_notes.txt
    ├── engineering_handbook.txt            # fictional demo doc for the eval
    └── engineering_handbook_questions.json
```

## Possible extensions

- Add an approximate FAISS index (HNSW/IVF) and measure the recall/latency trade-off
- Add hybrid search (BM25 + vectors) and compare it in the eval harness
- Add an LLM-based answer-quality and hallucination eval on top of the retrieval eval
- Swap in a hosted vector DB (Pinecone/Weaviate) for multi-user scale
- Add source citations in the UI (which chunk/doc an answer came from)
- Add conversation memory (multi-turn context, not just single-shot Q&A)
- Containerize with Docker + docker-compose for one-command startup
