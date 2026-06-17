# GitBrain — AI-Powered GitHub Repository Intelligence

> Ask natural language questions about any GitHub repository and receive cited, accurate answers powered by Llama 3 via Groq.

---

## Project Overview

GitBrain is a full-stack **Retrieval-Augmented Generation (RAG)** application. It indexes any GitHub repository and lets developers query it in plain English through a React chat interface backed by a FastAPI + Groq AI engine.

**The problem it solves:** Developers waste hours reading unfamiliar codebases. GitBrain lets you ask _"how does authentication work?"_ and instantly get a cited answer pointing to the exact files and line numbers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend (Week 5)                       │
│  Vite · Tailwind CSS · Axios · Lucide React                     │
│  http://localhost:5173                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (Axios)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Week 4)                       │
│  GET /health   POST /query                                      │
│  http://127.0.0.1:8000                                          │
└───────────┬──────────────────────────────┬──────────────────────┘
            │                              │
            ▼                              ▼
┌───────────────────────┐    ┌─────────────────────────────────────┐
│   RAG Engine          │    │   ChromaDB Vector Index (Week 3)     │
│   rag_engine.py       │    │   data/chroma_db/                    │
│   llm_client.py       │    │   sentence-transformers embeddings   │
│   embedder.py         │    └─────────────────────────────────────┘
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Groq API / Llama 3  │
│   llama3-8b-8192       │
│   (free, no credit    │
│    card needed)       │
└───────────────────────┘
```

---

## Week-by-Week Summary

| Week       | Focus               | Key Deliverables                                          |
| ---------- | ------------------- | --------------------------------------------------------- |
| **Week 1** | GitHub Ingestion    | Fetch files, language detection, AST chunking             |
| **Week 2** | Data Quality        | JS/TS heuristic chunker, JSONL export, validation, pytest |
| **Week 3** | Vector Intelligence | sentence-transformers, ChromaDB, semantic search CLI      |
| **Week 4** | FastAPI Backend     | RAG engine, Groq/Llama 3, `/health` + `/query` endpoints  |
| **Week 5** | React Frontend      | Chat UI, source citations, repo setup, status sidebar     |

---

## Folder Structure

```
gitbrain/
├── frontend/                    ← React app (Week 5)
│   ├── src/
│   │   ├── App.jsx              ← Root component, global state
│   │   ├── services/
│   │   │   └── apiClient.js     ← All HTTP calls to FastAPI
│   │   ├── components/
│   │   │   ├── Sidebar.jsx      ← Status indicators, repo info
│   │   │   ├── ChatWindow.jsx   ← Chat input + message list
│   │   │   ├── ChatMessage.jsx  ← Single message bubble
│   │   │   ├── SourcePanel.jsx  ← Collapsible citations
│   │   │   ├── RepoSetup.jsx    ← GitHub URL input + ingestion
│   │   │   ├── StatusCard.jsx   ← Status badge component
│   │   │   └── LoadingDots.jsx  ← Animated loading indicator
│   │   └── utils/
│   │       └── repoUtils.js     ← URL parsing, score formatting
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── .env.example
├── api/                         ← FastAPI application (Week 4)
│   ├── main.py                  ← App, CORS, startup
│   └── routes/
│       ├── health.py            ← GET /health
│       └── query.py             ← POST /query
├── core/                        ← AI/ML pipeline
│   ├── rag_engine.py            ← RAG orchestrator (Week 4)
│   ├── llm_client.py            ← Groq wrapper (Week 4)
│   ├── embedder.py              ← sentence-transformers (Week 3)
│   ├── vector_store.py          ← ChromaDB wrapper (Week 3)
│   ├── code_chunker.py          ← Code chunking (Week 2)
│   └── github_ingester.py       ← GitHub API (Week 1)
├── config/
│   └── settings.py              ← Pydantic BaseSettings (Week 4)
├── utils/
│   ├── repo_parser.py           ← URL parsing (Week 1)
│   └── file_utils.py            ← JSONL helpers (Week 2)
├── scripts/
│   ├── export_chunks.py         ← Fetch repo → JSONL (Week 2)
│   ├── validate_chunks.py       ← Quality checks (Week 2)
│   ├── build_vector_index.py    ← JSONL → ChromaDB (Week 3)
│   └── semantic_search.py       ← CLI search (Week 3)
├── tests/
│   ├── test_frontend_api.py     ← 28 API tests for frontend (Week 5)
│   ├── test_api.py              ← 22 FastAPI tests (Week 4)
│   ├── test_vector_store.py     ← 14 ChromaDB tests (Week 3)
│   ├── test_embedder.py         ← 13 embedding tests (Week 3)
│   ├── test_chunker.py          ← 9 chunking tests (Week 2)
│   └── test_repo_parser.py      ← 8 URL parsing tests (Week 2)
├── data/
│   ├── chunks/                  ← JSONL files
│   └── chroma_db/               ← Vector index
├── requirements.txt
└── .env.example
```

---

## Backend Setup (Python)

```bat
:: Windows
cd gitbrain
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
GITHUB_TOKEN=ghp_your_token_here
```

Get a **free Groq key** at [console.groq.com](https://console.groq.com) (no credit card).

---

## Frontend Setup (React)

```bat
cd gitbrain\frontend
npm install
copy .env.example .env
```

The frontend `.env` only needs one line:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Running Both Services

**You need two terminals open simultaneously.**

**Terminal 1 — Backend:**

```bat
cd gitbrain
venv\Scripts\activate
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```bat
cd gitbrain\frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Index a Repository (One-Time Setup)

Before you can chat, you need to index a repository:

```bat
:: In Terminal 1 (with venv activated)
python scripts\export_chunks.py https://github.com/psf/requests
python scripts\build_vector_index.py data\chunks\psf__requests_chunks.jsonl
```

Then in the React app:

1. Enter `https://github.com/psf/requests` in the Repository Setup panel
2. Click the terminal icon (manual set) — this sets the repo without re-indexing
3. Start chatting!

---

## Environment Variables

### Backend (`gitbrain/.env`)

| Variable               | Required    | Default            | Description                             |
| ---------------------- | ----------- | ------------------ | --------------------------------------- |
| `GROQ_API_KEY`         | **Yes**     | —                  | Groq API key (free at console.groq.com) |
| `GITHUB_TOKEN`         | Recommended | —                  | GitHub token (5,000 req/hr vs 60)       |
| `LLM_MODEL`            | No          | `llama3-8b-8192`   | Groq model identifier                   |
| `EMBEDDING_MODEL`      | No          | `all-MiniLM-L6-v2` | sentence-transformers model             |
| `CHROMA_PERSIST_DIR`   | No          | `data/chroma_db`   | ChromaDB storage path                   |
| `TOP_K_RESULTS`        | No          | `5`                | Chunks to retrieve per query            |
| `SIMILARITY_THRESHOLD` | No          | `0.35`             | Min cosine similarity score             |

### Frontend (`gitbrain/frontend/.env`)

| Variable            | Required | Default                 | Description         |
| ------------------- | -------- | ----------------------- | ------------------- |
| `VITE_API_BASE_URL` | No       | `http://127.0.0.1:8000` | FastAPI backend URL |

---

## Run Tests

```bat
cd gitbrain
venv\Scripts\activate

:: All offline tests (no API key needed, ~8 seconds)
pytest tests\ -v

:: Frontend API tests only
pytest tests\test_frontend_api.py -v
```

**Expected: 82 tests, 0 failures.**

---

## API Reference

### GET /health

```json
{
  "status": "ok",
  "service": "GitBrain API",
  "groq_key": "configured",
  "chroma": "data/chroma_db"
}
```

### POST /query

**Request:**

```json
{
  "question": "how does authentication work?",
  "repo_name": "psf__requests",
  "top_k": 5
}
```

**Response:**

```json
{
  "answer": "Authentication is handled by HTTPBasicAuth in requests/auth.py...",
  "sources": [{ "file": "requests/auth.py", "lines": "1-50", "score": 0.89 }],
  "chunks_retrieved": 3,
  "repo_name": "psf__requests"
}
```

---

## Troubleshooting

| Problem                        | Fix                                                               |
| ------------------------------ | ----------------------------------------------------------------- |
| `npm` not recognized           | Install Node.js from nodejs.org (LTS version)                     |
| Sidebar shows backend offline  | Start uvicorn in Terminal 1 first                                 |
| CORS errors in browser         | Check `VITE_API_BASE_URL` in `frontend/.env` matches backend port |
| `GROQ_API_KEY not set`         | Add key to `gitbrain/.env`, restart uvicorn                       |
| Collection not found (404)     | Run `build_vector_index.py` first, then set repo manually         |
| Slow first query (10–15s)      | Embedding model downloading once — subsequent queries are fast    |
| `npm install` permission error | Run terminal as Administrator                                     |
| Port 5173 already in use       | `npm run dev -- --port 5174`                                      |
| Port 8000 already in use       | `uvicorn api.main:app --port 8001`                                |
