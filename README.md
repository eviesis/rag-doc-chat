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
│   │   ├── embeddings.py    # local sentence-transformers wrapper
│   │   ├── vectorstore.py   # Chroma wrapper
│   │   └── llm_client.py    # Claude client: streaming + retry + rate limit
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
    └── sample_notes.txt
```

## Possible extensions

- Swap Chroma for a hosted vector DB (Pinecone/Weaviate) for multi-user scale
- Add source citations in the UI (which chunk/doc an answer came from)
- Add conversation memory (multi-turn context, not just single-shot Q&A)
- Containerize with Docker + docker-compose for one-command startup
