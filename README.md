# RAG System

A Retrieval-Augmented Generation (RAG) system built with Python, Qdrant, PostgreSQL, and LiteLLM.

## Features

- **Document Ingestion**: PDF, text files, code files, web pages
- **Vector Search**: Fast similarity search with Qdrant
- **Multiple LLMs**: Access OpenAI, Anthropic, and more through LiteLLM
- **REST API**: FastAPI server for easy integration
- **OpenWebUI Compatible**: Works with OpenWebUI pipelines

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your NAS                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Qdrant  │  │ Postgres │  │ LiteLLM  │  │ OpenWebUI│    │
│  │  :6333   │  │  :5432   │  │  :4000   │  │  :3000   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↑
                        Python RAG Code
```

## Quick Start

### 1. Set up NAS (Docker)

Copy the files from `docker/` folder to your NAS and run:

```bash
cd /volume1/docker/rag-system/
docker-compose up -d
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your NAS IP and API keys
```

### 3. Install Dependencies

```bash
uv sync  # or: pip install -e .
```

### 4. Initialize System

```python
from ingestion.pipeline import initialize_system
initialize_system()
```

### 5. Ingest Documents

```python
from ingestion.pipeline import ingest_document

# Ingest a PDF
doc_id = ingest_document("path/to/document.pdf")

# Ingest a web page
doc_id = ingest_document("https://example.com/article")

# Ingest a directory
from ingestion.pipeline import ingest_directory
doc_ids = ingest_directory("./documents", extensions=[".pdf", ".txt"])
```

### 6. Query

```python
from retrieval.retriever import rag_query

result = rag_query("What is this document about?")
print(result["answer"])
print(result["sources"])
```

## API Server

Start the server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | Query with RAG |
| POST | `/context` | Get context without LLM |
| POST | `/ingest/file` | Upload document |
| POST | `/ingest/url` | Ingest web page |
| GET | `/documents` | List documents |
| DELETE | `/documents/{id}` | Delete document |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 3}'

# Ingest URL
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/RAG"}'
```

## Project Structure

```
RAG-System/
├── config.py              # Configuration
├── database/              # Database connections
│   ├── postgres.py        # PostgreSQL models
│   └── qdrant.py          # Qdrant operations
├── ingestion/             # Document processing
│   ├── loaders.py         # File loaders
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Embedding generation
│   └── pipeline.py        # Ingestion pipeline
├── retrieval/             # RAG retrieval
│   └── retriever.py       # Search and prompt building
├── api/                   # REST API
│   └── server.py          # FastAPI server
├── docker/                # NAS deployment
│   ├── docker-compose.yml
│   └── litellm_config.yaml
└── TUTORIAL.md            # Step-by-step guide
```

## Configuration

All settings are in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `NAS_IP` | Your NAS IP address | `` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `POSTGRES_PASSWORD` | Database password | - |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `CHUNK_SIZE` | Tokens per chunk | `500` |
| `RAG_TOP_K` | Chunks to retrieve | `5` |

## Tutorial

For a complete step-by-step guide, see [TUTORIAL.md](TUTORIAL.md).

## License

MIT
