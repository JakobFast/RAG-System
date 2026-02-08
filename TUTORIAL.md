# RAG System Tutorial

A complete guide to building a Retrieval-Augmented Generation (RAG) system on your NAS with Python.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              YOUR NAS                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Qdrant    │  │ PostgreSQL  │  │   LiteLLM   │  │  OpenWebUI  │    │
│  │   :6333     │  │   :5432     │  │   :4000     │  │   :3000     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Network Connection
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           YOUR LOCAL PC                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Python RAG Code (development)                                   │    │
│  │  - Document ingestion                                            │    │
│  │  - Embedding generation                                          │    │
│  │  - Retrieval logic                                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: NAS Docker Setup

### Step 1.1: Create Project Folder on NAS

- [ ] SSH into your NAS or use File Station
- [ ] Create the folder structure:

```bash
# On your NAS (adjust path for your NAS model)
mkdir -p /volume1/docker/rag-system
cd /volume1/docker/rag-system

# Create data directories
mkdir -p qdrant_data postgres_data openwebui_data
```

### Step 1.2: Create Environment File on NAS

- [ ] Create `.env` file on NAS at `/volume1/docker/rag-system/.env`:

```env
# ===========================================
# NAS Environment Variables
# ===========================================

# OpenAI API Key - Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-your-openai-key-here

# Anthropic API Key - Get from https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# PostgreSQL Password - Choose a secure password
POSTGRES_PASSWORD=your-secure-password-here

# LiteLLM Master Key (optional, for admin access)
LITELLM_MASTER_KEY=sk-litellm-master-key-here
```

### Step 1.3: Create docker-compose.yml on NAS

- [ ] Create `docker-compose.yml` at `/volume1/docker/rag-system/docker-compose.yml`:

```yaml
version: '3.8'

services:
  # ===========================================
  # QDRANT - Vector Database
  # ===========================================
  # Stores embeddings (numerical representations of text)
  # Dashboard: http://NAS-IP:6333/dashboard
  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-qdrant
    ports:
      - "6333:6333"    # REST API
      - "6334:6334"    # gRPC (faster protocol)
    volumes:
      - ./qdrant_data:/qdrant/storage
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ===========================================
  # POSTGRESQL - Relational Database
  # ===========================================
  # Stores document metadata, chunk text, relationships
  postgres:
    image: postgres:16
    container_name: rag-postgres
    environment:
      POSTGRES_DB: rag_system
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_system"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ===========================================
  # LITELLM - LLM Proxy
  # ===========================================
  # Unified API for OpenAI, Anthropic, and other LLMs
  # API: http://NAS-IP:4000
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    container_name: rag-litellm
    ports:
      - "4000:4000"
    volumes:
      - ./litellm_config.yaml:/app/config.yaml
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}
    command: ["--config", "/app/config.yaml"]
    restart: unless-stopped
    depends_on:
      - postgres

  # ===========================================
  # OPEN-WEBUI - Chat Interface
  # ===========================================
  # Beautiful chat UI that connects to LiteLLM
  # UI: http://NAS-IP:3000
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: rag-openwebui
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://litellm:4000/v1
      - OPENAI_API_KEY=${LITELLM_MASTER_KEY:-dummy-key}
      - WEBUI_AUTH=true
    volumes:
      - ./openwebui_data:/app/backend/data
    depends_on:
      - litellm
    restart: unless-stopped

networks:
  default:
    name: rag-network
```

### Step 1.4: Create LiteLLM Config on NAS

- [ ] Create `litellm_config.yaml` at `/volume1/docker/rag-system/litellm_config.yaml`:

```yaml
# ===========================================
# LiteLLM Configuration
# ===========================================
# Defines which LLM models are available through the proxy

model_list:
  # -----------------------------------------
  # OpenAI Models
  # -----------------------------------------
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4-turbo
    litellm_params:
      model: openai/gpt-4-turbo
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

  # -----------------------------------------
  # Anthropic Models
  # -----------------------------------------
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-haiku
    litellm_params:
      model: anthropic/claude-3-haiku-20240307
      api_key: os.environ/ANTHROPIC_API_KEY

  # -----------------------------------------
  # Embedding Models (for RAG)
  # -----------------------------------------
  - model_name: text-embedding-3-small
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY

  - model_name: text-embedding-3-large
    litellm_params:
      model: openai/text-embedding-3-large
      api_key: os.environ/OPENAI_API_KEY

# General settings
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY

# Logging (optional)
litellm_settings:
  drop_params: true
  set_verbose: false
```

### Step 1.5: Start Docker Containers

- [ ] SSH into your NAS and run:

```bash
cd /volume1/docker/rag-system

# Start all containers in detached mode
docker-compose up -d

# Check if all containers are running
docker-compose ps

# View logs (optional, useful for debugging)
docker-compose logs -f
```

### Step 1.6: Verify Services are Running

- [ ] Test Qdrant: Open `http://YOUR-NAS-IP:6333/dashboard` in browser
- [ ] Test PostgreSQL: Connect with a database client (DBeaver, pgAdmin, etc.)
  - Host: `YOUR-NAS-IP`
  - Port: `5432`
  - Database: `rag_system`
  - User: `rag_user`
  - Password: (from your .env)
- [ ] Test LiteLLM: Open `http://YOUR-NAS-IP:4000/health` - should show `{"status":"healthy"}`
- [ ] Test OpenWebUI: Open `http://YOUR-NAS-IP:3000` - create an account

---

## Part 2: Local Python Development Setup

### Step 2.1: Create .env File Locally

- [ ] Create `.env` in your project folder (`F:\Programmieren\Python\RAG-System\.env`):

```env
# ===========================================
# Local Development Environment Variables
# ===========================================

# Your NAS IP Address (CHANGE THIS!)
NAS_IP=192.168.1.100

# API Keys (same as NAS, or use different ones for dev)
OPENAI_API_KEY=sk-proj-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# PostgreSQL (must match NAS .env)
POSTGRES_PASSWORD=your-secure-password-here
POSTGRES_USER=rag_user
POSTGRES_DB=rag_system

# Embedding settings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Chunking settings
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

### Step 2.2: Create .env.example (Safe to Commit)

- [ ] Create `.env.example` as a template for others:

```env
# ===========================================
# Environment Variables Template
# ===========================================
# Copy this file to .env and fill in your values

# Your NAS IP Address
NAS_IP=192.168.x.x

# API Keys - Get from respective providers
OPENAI_API_KEY=sk-proj-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here

# PostgreSQL credentials
POSTGRES_PASSWORD=choose-a-secure-password
POSTGRES_USER=rag_user
POSTGRES_DB=rag_system

# Embedding settings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Chunking settings
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

### Step 2.3: Install Additional Dependencies

- [ ] Add missing packages to your project:

```bash
# Using uv (your package manager)
uv add pypdf trafilatura chardet python-dotenv

# Or with pip
pip install pypdf trafilatura chardet python-dotenv
```

### Step 2.4: Create config.py

- [ ] Create `config.py` in your project root:

```python
# config.py
"""
Centralized configuration for the RAG system.
Loads settings from environment variables.
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ===========================================
# NAS Connection Settings
# ===========================================
NAS_IP = os.getenv("NAS_IP", "192.168.1.100")

# Qdrant Vector Database
QDRANT_HOST = NAS_IP
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

# PostgreSQL Database
POSTGRES_HOST = NAS_IP
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_system")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# SQLAlchemy connection string
DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# LiteLLM Proxy
LITELLM_BASE_URL = f"http://{NAS_IP}:4000/v1"

# ===========================================
# API Keys
# ===========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ===========================================
# Embedding Settings
# ===========================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# ===========================================
# Chunking Settings
# ===========================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # overlap tokens

# ===========================================
# RAG Settings
# ===========================================
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # number of chunks to retrieve
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.7"))
```

---

## Part 3: Database Layer

### Step 3.1: Create Database Folder Structure

- [ ] Create the folder:

```bash
mkdir database
```

### Step 3.2: Create database/__init__.py

- [ ] Create `database/__init__.py`:

```python
# database/__init__.py
"""Database connections and models."""
from database.postgres import (
    Document,
    Chunk,
    Collection,
    init_database,
    get_session,
    SessionLocal,
)
from database.qdrant import (
    init_collection,
    store_embeddings,
    search_similar,
    delete_by_document,
)

__all__ = [
    "Document",
    "Chunk",
    "Collection",
    "init_database",
    "get_session",
    "SessionLocal",
    "init_collection",
    "store_embeddings",
    "search_similar",
    "delete_by_document",
]
```

### Step 3.3: Create database/postgres.py

- [ ] Create `database/postgres.py`:

```python
# database/postgres.py
"""
PostgreSQL database models and connection.

Uses SQLAlchemy ORM for easier database operations.
Tables:
- documents: Source files (PDF, text, web pages)
- chunks: Text segments with embeddings
- collections: Organize documents into groups
"""
import uuid
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from config import DATABASE_URL

# ===========================================
# Database Connection
# ===========================================

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True to see SQL queries
    pool_size=5,
    max_overflow=10,
)

# Base class for all models
Base = declarative_base()

# Session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ===========================================
# Models
# ===========================================


class Document(Base):
    """
    Represents a source document (PDF, text file, web page, etc.)
    One document can have many chunks.
    """

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(50))  # pdf, txt, md, html, py, etc.
    file_path = Column(Text)  # Original path or URL
    file_hash = Column(String(64), unique=True)  # SHA256 for deduplication
    metadata = Column(JSONB, default=dict)  # Flexible additional data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship: document.chunks returns all chunks
    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document {self.filename} ({self.file_type})>"


class Chunk(Base):
    """
    A chunk is a segment of a document that gets embedded.
    Stores the actual text and links to the vector in Qdrant.
    """

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    chunk_index = Column(Integer)  # Position in document (0, 1, 2, ...)
    content = Column(Text, nullable=False)  # The actual text
    token_count = Column(Integer)  # Number of tokens
    metadata = Column(JSONB, default=dict)  # Page number, section, etc.
    qdrant_point_id = Column(UUID(as_uuid=True))  # ID in Qdrant
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship: chunk.document returns the parent document
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk {self.chunk_index} of {self.document_id}>"


class Collection(Base):
    """
    Collections let you organize documents into groups.
    E.g., "Work Documents", "Research Papers", "Code Docs"
    """

    __tablename__ = "collections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Collection {self.name}>"


# ===========================================
# Database Operations
# ===========================================


def init_database() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    print("PostgreSQL tables created successfully!")


def get_session() -> Generator[Session, None, None]:
    """
    Get a database session. Use with 'with' statement or as dependency.

    Example:
        with next(get_session()) as session:
            docs = session.query(Document).all()
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_document_by_hash(session: Session, file_hash: str) -> Document | None:
    """Check if a document with this hash already exists."""
    return session.query(Document).filter_by(file_hash=file_hash).first()


def get_all_documents(session: Session) -> list[Document]:
    """Get all documents."""
    return session.query(Document).order_by(Document.created_at.desc()).all()


def get_document_chunks(session: Session, document_id: uuid.UUID) -> list[Chunk]:
    """Get all chunks for a document."""
    return (
        session.query(Chunk)
        .filter_by(document_id=document_id)
        .order_by(Chunk.chunk_index)
        .all()
    )
```

### Step 3.4: Create database/qdrant.py

- [ ] Create `database/qdrant.py`:

```python
# database/qdrant.py
"""
Qdrant vector database operations.

Qdrant stores embeddings (vectors) and enables fast similarity search.
Each vector is linked to a chunk in PostgreSQL via the point ID.
"""
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_DIMENSIONS

# ===========================================
# Client Connection
# ===========================================

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ===========================================
# Collection Management
# ===========================================


def init_collection() -> None:
    """
    Create the Qdrant collection if it doesn't exist.

    A collection is like a table - it holds all vectors with the same dimensions.
    """
    collections = client.get_collections().collections
    exists = any(c.name == QDRANT_COLLECTION for c in collections)

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,  # Must match embedding model output
                distance=Distance.COSINE,  # Cosine similarity (most common for text)
            ),
        )
        print(f"Created Qdrant collection: {QDRANT_COLLECTION}")
    else:
        print(f"Qdrant collection '{QDRANT_COLLECTION}' already exists")


def delete_collection() -> None:
    """Delete the collection (use with caution!)."""
    client.delete_collection(collection_name=QDRANT_COLLECTION)
    print(f"Deleted Qdrant collection: {QDRANT_COLLECTION}")


def get_collection_info() -> dict:
    """Get information about the collection."""
    info = client.get_collection(collection_name=QDRANT_COLLECTION)
    return {
        "name": QDRANT_COLLECTION,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status,
    }


# ===========================================
# Vector Operations
# ===========================================


def store_embeddings(
    embeddings: list[list[float]],
    chunk_ids: list[uuid.UUID],
    payloads: list[dict],
) -> None:
    """
    Store embeddings in Qdrant.

    Args:
        embeddings: List of vectors (each is a list of floats)
        chunk_ids: UUIDs that link back to PostgreSQL chunks
        payloads: Metadata stored with each vector (for filtering)
    """
    points = [
        PointStruct(
            id=str(chunk_id),  # Qdrant uses string IDs
            vector=embedding,
            payload=payload,
        )
        for chunk_id, embedding, payload in zip(chunk_ids, embeddings, payloads)
    ]

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search_similar(
    query_embedding: list[float],
    top_k: int = 5,
    score_threshold: float = 0.7,
    filter_conditions: Optional[dict] = None,
) -> list[dict]:
    """
    Find chunks similar to the query.

    Args:
        query_embedding: Vector representation of the query
        top_k: Number of results to return
        score_threshold: Minimum similarity score (0-1 for cosine)
        filter_conditions: Optional filters (e.g., {"file_type": "pdf"})

    Returns:
        List of matches with id, score, and payload
    """
    # Build filter if provided
    qdrant_filter = None
    if filter_conditions:
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filter_conditions.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=qdrant_filter,
    )

    return [
        {
            "chunk_id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results
    ]


def delete_by_document(document_id: uuid.UUID) -> None:
    """Delete all vectors belonging to a document."""
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id)),
                )
            ]
        ),
    )


def delete_points(point_ids: list[str]) -> None:
    """Delete specific points by ID."""
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=point_ids,
    )
```

---

## Part 4: Document Ingestion Layer

### Step 4.1: Create Ingestion Folder Structure

- [ ] Create the folder:

```bash
mkdir ingestion
```

### Step 4.2: Create ingestion/__init__.py

- [ ] Create `ingestion/__init__.py`:

```python
# ingestion/__init__.py
"""Document ingestion pipeline."""
from ingestion.loaders import load_document
from ingestion.chunker import chunk_text, count_tokens
from ingestion.embedder import generate_embeddings, generate_single_embedding
from ingestion.pipeline import ingest_document, initialize_system

__all__ = [
    "load_document",
    "chunk_text",
    "count_tokens",
    "generate_embeddings",
    "generate_single_embedding",
    "ingest_document",
    "initialize_system",
]
```

### Step 4.3: Create ingestion/loaders.py

- [ ] Create `ingestion/loaders.py`:

```python
# ingestion/loaders.py
"""
Document loaders for different file types.

Supported formats:
- PDF (.pdf)
- Text files (.txt, .md)
- Code files (.py, .js, .ts, .java, .c, .cpp, .go, .rs, etc.)
- Web pages (http://, https://)
"""
import hashlib
from pathlib import Path
from typing import Tuple

import chardet

# PDF loading
from pypdf import PdfReader

# Web page loading
import trafilatura


def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash for deduplication."""
    return hashlib.sha256(content).hexdigest()


def load_pdf(file_path: str) -> Tuple[str, dict]:
    """
    Extract text from PDF file.

    Returns:
        Tuple of (text_content, metadata_dict)
    """
    reader = PdfReader(file_path)

    # Extract text from all pages
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"[Page {i + 1}]\n{page_text}")

    text = "\n\n".join(text_parts)

    # Extract metadata
    metadata = {
        "page_count": len(reader.pages),
        "pdf_metadata": dict(reader.metadata) if reader.metadata else {},
    }

    return text, metadata


def load_text_file(file_path: str) -> Tuple[str, dict]:
    """
    Load plain text file with automatic encoding detection.

    Works for .txt, .md, .py, .js, .ts, .java, etc.
    """
    path = Path(file_path)

    # Read raw bytes and detect encoding
    raw_bytes = path.read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected["encoding"] or "utf-8"

    text = raw_bytes.decode(encoding, errors="replace")

    metadata = {
        "encoding": encoding,
        "size_bytes": len(raw_bytes),
        "confidence": detected.get("confidence", 0),
    }

    return text, metadata


def load_web_page(url: str) -> Tuple[str, dict]:
    """
    Extract clean text from a web page.

    Uses trafilatura to remove navigation, ads, boilerplate, etc.
    """
    # Download and extract
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not download: {url}")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
    )
    if not text:
        raise ValueError(f"Could not extract text from: {url}")

    # Get metadata
    metadata_obj = trafilatura.extract_metadata(downloaded)
    metadata = {
        "title": metadata_obj.title if metadata_obj else None,
        "author": metadata_obj.author if metadata_obj else None,
        "date": metadata_obj.date if metadata_obj else None,
        "url": url,
    }

    return text, metadata


def load_document(file_path: str) -> Tuple[str, str, dict, str]:
    """
    Main entry point - detect file type and load appropriately.

    Args:
        file_path: Path to file or URL

    Returns:
        Tuple of (text, file_type, metadata, file_hash)
    """
    # Check if it's a URL
    if file_path.startswith(("http://", "https://")):
        text, metadata = load_web_page(file_path)
        file_hash = calculate_file_hash(text.encode("utf-8"))
        return text, "web", metadata, file_hash

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    # Calculate hash from file content
    file_hash = calculate_file_hash(path.read_bytes())

    # Route to appropriate loader
    if suffix == ".pdf":
        text, metadata = load_pdf(file_path)
        return text, "pdf", metadata, file_hash

    # Text-based files (code, markdown, plain text)
    text_extensions = {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".r",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".yaml",
        ".yml",
        ".json",
        ".xml",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".less",
    }

    if suffix in text_extensions:
        text, metadata = load_text_file(file_path)
        file_type = suffix[1:]  # Remove the dot
        return text, file_type, metadata, file_hash

    # Try as text file for unknown extensions
    try:
        text, metadata = load_text_file(file_path)
        return text, "unknown", metadata, file_hash
    except Exception as e:
        raise ValueError(f"Could not load file {file_path}: {e}")
```

### Step 4.4: Create ingestion/chunker.py

- [ ] Create `ingestion/chunker.py`:

```python
# ingestion/chunker.py
"""
Text chunking strategies.

Chunks should be:
- Small enough to embed well (500-1000 tokens)
- Large enough to contain meaningful context
- Overlapping to preserve context at boundaries
"""
import re
from typing import List, Tuple

import tiktoken

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Load tokenizer (cl100k_base is used by text-embedding-3-small/large)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in text."""
    return len(_tokenizer.encode(text))


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Tuple[str, int]]:
    """
    Split text into overlapping chunks.

    Strategy:
    1. Split by paragraphs first (preserve natural boundaries)
    2. Combine paragraphs until chunk_size is reached
    3. Add overlap from previous chunk

    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks

    Returns:
        List of (chunk_text, token_count) tuples
    """
    if not text.strip():
        return []

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[Tuple[str, int]] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Handle paragraphs larger than chunk_size
        if para_tokens > chunk_size:
            # First, save current chunk if any
            if current_parts:
                chunk_text_content = "\n\n".join(current_parts)
                chunks.append((chunk_text_content, current_tokens))
                current_parts = []
                current_tokens = 0

            # Split large paragraph by sentences
            sentences = _split_into_sentences(para)
            sent_parts: List[str] = []
            sent_tokens = 0

            for sentence in sentences:
                s_tokens = count_tokens(sentence)

                if sent_tokens + s_tokens > chunk_size and sent_parts:
                    # Save sentence chunk
                    chunk_text_content = " ".join(sent_parts)
                    chunks.append((chunk_text_content, sent_tokens))

                    # Calculate overlap
                    overlap_text = chunk_text_content[
                        -chunk_overlap * 4 :
                    ]  # Approximate
                    overlap_tokens = count_tokens(overlap_text)

                    sent_parts = [overlap_text] if overlap_text else []
                    sent_tokens = overlap_tokens

                sent_parts.append(sentence)
                sent_tokens += s_tokens

            # Don't forget remaining sentences
            if sent_parts:
                chunk_text_content = " ".join(sent_parts)
                chunks.append((chunk_text_content, sent_tokens))

        # Normal case: check if adding paragraph exceeds limit
        elif current_tokens + para_tokens > chunk_size and current_parts:
            # Save current chunk
            chunk_text_content = "\n\n".join(current_parts)
            chunks.append((chunk_text_content, current_tokens))

            # Calculate overlap from current chunk
            overlap_parts: List[str] = []
            overlap_tokens = 0

            for part in reversed(current_parts):
                part_tokens = count_tokens(part)
                if overlap_tokens + part_tokens <= chunk_overlap:
                    overlap_parts.insert(0, part)
                    overlap_tokens += part_tokens
                else:
                    break

            # Start new chunk with overlap + new paragraph
            current_parts = overlap_parts + [para]
            current_tokens = overlap_tokens + para_tokens

        else:
            # Add paragraph to current chunk
            current_parts.append(para)
            current_tokens += para_tokens

    # Don't forget the last chunk!
    if current_parts:
        chunk_text_content = "\n\n".join(current_parts)
        chunks.append((chunk_text_content, current_tokens))

    return chunks


def chunk_code(
    code: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Tuple[str, int]]:
    """
    Chunk code while trying to preserve function/class boundaries.

    This is a simple implementation - for production, consider using
    tree-sitter or similar for proper AST-based chunking.
    """
    # Split by common code boundaries
    # This regex looks for function/class definitions
    boundaries = re.split(
        r"(\n(?=def |class |function |const |let |var |public |private |async ))",
        code,
    )

    chunks: List[Tuple[str, int]] = []
    current_parts: List[str] = []
    current_tokens = 0

    for part in boundaries:
        if not part.strip():
            continue

        part_tokens = count_tokens(part)

        if current_tokens + part_tokens > chunk_size and current_parts:
            chunk_text_content = "".join(current_parts)
            chunks.append((chunk_text_content, current_tokens))
            current_parts = []
            current_tokens = 0

        current_parts.append(part)
        current_tokens += part_tokens

    if current_parts:
        chunk_text_content = "".join(current_parts)
        chunks.append((chunk_text_content, current_tokens))

    return chunks
```

### Step 4.5: Create ingestion/embedder.py

- [ ] Create `ingestion/embedder.py`:

```python
# ingestion/embedder.py
"""
Generate embeddings using OpenAI via LiteLLM.

Embeddings convert text to vectors (lists of numbers) that
capture semantic meaning. Similar texts have similar vectors.
"""
import time
from typing import List

from openai import OpenAI

from config import LITELLM_BASE_URL, EMBEDDING_MODEL

# ===========================================
# Client Setup
# ===========================================

# Create client pointing to LiteLLM proxy on NAS
client = OpenAI(
    base_url=LITELLM_BASE_URL,
    api_key="dummy",  # LiteLLM handles actual API keys
)


# ===========================================
# Embedding Functions
# ===========================================


def generate_embeddings(
    texts: List[str],
    batch_size: int = 100,
    retry_delay: float = 1.0,
    max_retries: int = 3,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        batch_size: Process this many texts at once (API limit is ~2000)
        retry_delay: Seconds to wait between retries
        max_retries: Maximum number of retry attempts

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )

                # Extract embeddings in correct order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Embedding error (attempt {attempt + 1}): {e}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise

        # Rate limiting between batches
        if i + batch_size < len(texts):
            time.sleep(0.1)

    return all_embeddings


def generate_single_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text (used for queries).

    Args:
        text: The text to embed

    Returns:
        Embedding vector (list of floats)
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding
```

### Step 4.6: Create ingestion/pipeline.py

- [ ] Create `ingestion/pipeline.py`:

```python
# ingestion/pipeline.py
"""
Complete ingestion pipeline that ties everything together.

Flow:
1. Load document (extract text)
2. Check for duplicates
3. Chunk the text
4. Generate embeddings
5. Store in PostgreSQL + Qdrant
"""
import uuid
from pathlib import Path
from typing import Optional

from database.postgres import (
    Document,
    Chunk,
    SessionLocal,
    init_database,
    get_document_by_hash,
)
from database.qdrant import init_collection, store_embeddings
from ingestion.loaders import load_document
from ingestion.chunker import chunk_text, chunk_code
from ingestion.embedder import generate_embeddings


def initialize_system() -> None:
    """Initialize both databases. Call once at startup."""
    print("Initializing RAG system...")
    init_database()
    init_collection()
    print("System ready!")


def ingest_document(
    file_path: str,
    collection_name: Optional[str] = None,
    use_code_chunking: bool = False,
) -> uuid.UUID:
    """
    Ingest a document into the RAG system.

    Args:
        file_path: Path to file or URL
        collection_name: Optional collection to add document to
        use_code_chunking: Use code-aware chunking (for source files)

    Returns:
        Document UUID

    Raises:
        ValueError: If document loading fails
        Exception: If database operations fail
    """
    print(f"\n{'='*60}")
    print(f"Ingesting: {file_path}")
    print("=" * 60)

    # Step 1: Load document
    print("\n[1/6] Loading document...")
    text, file_type, metadata, file_hash = load_document(file_path)
    print(f"      Type: {file_type}")
    print(f"      Size: {len(text):,} characters")

    # Step 2: Check for duplicates
    print("\n[2/6] Checking for duplicates...")
    session = SessionLocal()
    try:
        existing = get_document_by_hash(session, file_hash)
        if existing:
            print(f"      Document already exists: {existing.id}")
            return existing.id

        # Step 3: Create document record
        print("\n[3/6] Creating document record...")
        filename = (
            Path(file_path).name if not file_path.startswith("http") else file_path
        )

        doc = Document(
            filename=filename,
            file_type=file_type,
            file_path=file_path,
            file_hash=file_hash,
            metadata=metadata,
        )
        session.add(doc)
        session.flush()  # Get the ID
        print(f"      Document ID: {doc.id}")

        # Step 4: Chunk the text
        print("\n[4/6] Chunking text...")
        if use_code_chunking or file_type in ["py", "js", "ts", "java", "c", "cpp", "go"]:
            chunks_data = chunk_code(text)
        else:
            chunks_data = chunk_text(text)
        print(f"      Created {len(chunks_data)} chunks")

        if not chunks_data:
            print("      WARNING: No chunks created (empty document?)")
            session.commit()
            return doc.id

        # Step 5: Generate embeddings
        print("\n[5/6] Generating embeddings...")
        chunk_texts = [c[0] for c in chunks_data]
        embeddings = generate_embeddings(chunk_texts)
        print(f"      Generated {len(embeddings)} embeddings")

        # Step 6: Store everything
        print("\n[6/6] Storing in databases...")

        chunk_records = []
        chunk_ids = []
        payloads = []

        for i, ((chunk_content, token_count), embedding) in enumerate(
            zip(chunks_data, embeddings)
        ):
            chunk_id = uuid.uuid4()

            chunk = Chunk(
                id=chunk_id,
                document_id=doc.id,
                chunk_index=i,
                content=chunk_content,
                token_count=token_count,
                qdrant_point_id=chunk_id,
                metadata={"index": i, "collection": collection_name},
            )
            chunk_records.append(chunk)
            chunk_ids.append(chunk_id)

            # Payload for Qdrant (enables filtering during search)
            payloads.append(
                {
                    "document_id": str(doc.id),
                    "document_name": doc.filename,
                    "file_type": file_type,
                    "chunk_index": i,
                    "collection": collection_name,
                }
            )

        session.add_all(chunk_records)

        # Store embeddings in Qdrant
        store_embeddings(embeddings, chunk_ids, payloads)
        print("      Stored in Qdrant")

        session.commit()
        print("      Stored in PostgreSQL")

        print(f"\n{'='*60}")
        print(f"SUCCESS: Document ingested with ID: {doc.id}")
        print("=" * 60)

        return doc.id

    except Exception as e:
        session.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        session.close()


def ingest_directory(
    directory_path: str,
    extensions: Optional[list[str]] = None,
    recursive: bool = True,
    collection_name: Optional[str] = None,
) -> list[uuid.UUID]:
    """
    Ingest all documents from a directory.

    Args:
        directory_path: Path to directory
        extensions: List of file extensions to include (e.g., [".pdf", ".txt"])
        recursive: Search subdirectories
        collection_name: Optional collection name

    Returns:
        List of document UUIDs
    """
    path = Path(directory_path)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {directory_path}")

    # Default extensions
    if extensions is None:
        extensions = [".pdf", ".txt", ".md", ".py", ".js", ".ts"]

    # Find files
    pattern = "**/*" if recursive else "*"
    files = [
        f for f in path.glob(pattern) if f.is_file() and f.suffix.lower() in extensions
    ]

    print(f"Found {len(files)} files to ingest")

    doc_ids = []
    for file in files:
        try:
            doc_id = ingest_document(str(file), collection_name=collection_name)
            doc_ids.append(doc_id)
        except Exception as e:
            print(f"Failed to ingest {file}: {e}")

    return doc_ids
```

---

## Part 5: Retrieval System

### Step 5.1: Create Retrieval Folder

- [ ] Create the folder:

```bash
mkdir retrieval
```

### Step 5.2: Create retrieval/__init__.py

- [ ] Create `retrieval/__init__.py`:

```python
# retrieval/__init__.py
"""RAG retrieval system."""
from retrieval.retriever import (
    retrieve_context,
    build_rag_prompt,
    rag_query,
)

__all__ = [
    "retrieve_context",
    "build_rag_prompt",
    "rag_query",
]
```

### Step 5.3: Create retrieval/retriever.py

- [ ] Create `retrieval/retriever.py`:

```python
# retrieval/retriever.py
"""
RAG retrieval - find relevant chunks and build context.

This is the core of the RAG system:
1. Convert query to embedding
2. Find similar chunks in Qdrant
3. Fetch full content from PostgreSQL
4. Build augmented prompt for LLM
"""
import uuid
from typing import List, Optional

from openai import OpenAI

from config import LITELLM_BASE_URL, RAG_TOP_K, RAG_SCORE_THRESHOLD
from database.postgres import Chunk, Document, SessionLocal
from database.qdrant import search_similar
from ingestion.embedder import generate_single_embedding

# LLM client for chat completions
llm_client = OpenAI(
    base_url=LITELLM_BASE_URL,
    api_key="dummy",
)


def retrieve_context(
    query: str,
    top_k: int = RAG_TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
    file_type_filter: Optional[str] = None,
    collection_filter: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        score_threshold: Minimum relevance score (0-1)
        file_type_filter: Filter by document type (e.g., "pdf")
        collection_filter: Filter by collection name

    Returns:
        List of chunks with content, metadata, and scores
    """
    # Generate query embedding
    query_embedding = generate_single_embedding(query)

    # Build filters
    filters = {}
    if file_type_filter:
        filters["file_type"] = file_type_filter
    if collection_filter:
        filters["collection"] = collection_filter

    # Search Qdrant
    results = search_similar(
        query_embedding=query_embedding,
        top_k=top_k,
        score_threshold=score_threshold,
        filter_conditions=filters if filters else None,
    )

    if not results:
        return []

    # Fetch full content from PostgreSQL
    session = SessionLocal()
    try:
        enriched_results = []
        for result in results:
            chunk = (
                session.query(Chunk).filter_by(id=uuid.UUID(result["chunk_id"])).first()
            )

            if chunk:
                doc = (
                    session.query(Document).filter_by(id=chunk.document_id).first()
                )
                enriched_results.append(
                    {
                        "content": chunk.content,
                        "score": result["score"],
                        "document_name": doc.filename if doc else "Unknown",
                        "document_type": doc.file_type if doc else "Unknown",
                        "document_path": doc.file_path if doc else None,
                        "chunk_index": chunk.chunk_index,
                        "document_id": str(chunk.document_id),
                        "token_count": chunk.token_count,
                    }
                )

        return enriched_results
    finally:
        session.close()


def build_rag_prompt(
    query: str,
    contexts: List[dict],
    system_prompt: Optional[str] = None,
) -> str:
    """
    Build a prompt with retrieved context.

    Args:
        query: User's question
        contexts: Retrieved context chunks
        system_prompt: Optional custom system prompt

    Returns:
        Augmented prompt for LLM
    """
    if not contexts:
        return query

    # Format context
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        source = ctx["document_name"]
        score = ctx["score"]
        content = ctx["content"]
        context_parts.append(
            f"[Source {i}: {source} (relevance: {score:.2f})]\n{content}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    # Build prompt
    prompt = f"""Use the following context to answer the question. If the context doesn't contain relevant information, say so and answer based on your general knowledge.

CONTEXT:
{context_text}

---

QUESTION: {query}

ANSWER:"""

    return prompt


def rag_query(
    query: str,
    model: str = "gpt-4o-mini",
    top_k: int = RAG_TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
    file_type_filter: Optional[str] = None,
    collection_filter: Optional[str] = None,
    include_sources: bool = True,
) -> dict:
    """
    Complete RAG query: retrieve context and generate answer.

    Args:
        query: User's question
        model: LLM model to use
        top_k: Number of chunks to retrieve
        score_threshold: Minimum relevance score
        file_type_filter: Filter by document type
        collection_filter: Filter by collection
        include_sources: Include source information in response

    Returns:
        Dict with 'answer', 'contexts', and 'model' used
    """
    # Retrieve relevant context
    contexts = retrieve_context(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        file_type_filter=file_type_filter,
        collection_filter=collection_filter,
    )

    # Build augmented prompt
    augmented_prompt = build_rag_prompt(query, contexts)

    # Generate answer
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Be accurate and cite your sources when relevant.",
            },
            {"role": "user", "content": augmented_prompt},
        ],
        temperature=0.7,
    )

    answer = response.choices[0].message.content

    result = {
        "answer": answer,
        "model": model,
        "contexts_used": len(contexts),
    }

    if include_sources:
        result["sources"] = [
            {
                "document": ctx["document_name"],
                "relevance": ctx["score"],
                "chunk_index": ctx["chunk_index"],
            }
            for ctx in contexts
        ]

    return result
```

---

## Part 6: API Server

### Step 6.1: Create API Folder

- [ ] Create the folder:

```bash
mkdir api
```

### Step 6.2: Create api/__init__.py

- [ ] Create `api/__init__.py`:

```python
# api/__init__.py
"""FastAPI server for RAG system."""
from api.server import app

__all__ = ["app"]
```

### Step 6.3: Create api/server.py

- [ ] Create `api/server.py`:

```python
# api/server.py
"""
FastAPI server exposing RAG functionality.

Endpoints:
- POST /query - Query the RAG system
- POST /ingest/file - Upload and ingest a file
- POST /ingest/url - Ingest a web page
- GET /documents - List all documents
- DELETE /documents/{id} - Delete a document
- GET /health - Health check
"""
import os
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ingestion.pipeline import ingest_document, initialize_system
from retrieval.retriever import retrieve_context, build_rag_prompt, rag_query
from database.postgres import Document, SessionLocal, get_all_documents
from database.qdrant import delete_by_document, get_collection_info

# ===========================================
# FastAPI App
# ===========================================

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API for document Q&A",
    version="1.0.0",
)

# Allow connections from OpenWebUI and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================
# Request/Response Models
# ===========================================


class QueryRequest(BaseModel):
    """Request for RAG query."""

    query: str = Field(..., description="The question to ask")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    score_threshold: float = Field(
        0.7, ge=0, le=1, description="Minimum relevance score"
    )
    file_type: Optional[str] = Field(None, description="Filter by document type")
    collection: Optional[str] = Field(None, description="Filter by collection")
    model: str = Field("gpt-4o-mini", description="LLM model to use")


class QueryResponse(BaseModel):
    """Response from RAG query."""

    answer: str
    model: str
    contexts_used: int
    sources: list[dict] = []


class ContextRequest(BaseModel):
    """Request for context retrieval (without LLM)."""

    query: str
    top_k: int = 5
    file_type: Optional[str] = None
    collection: Optional[str] = None


class ContextResponse(BaseModel):
    """Response with retrieved contexts."""

    augmented_prompt: str
    contexts: list[dict]


class IngestURLRequest(BaseModel):
    """Request to ingest a URL."""

    url: str = Field(..., description="URL to ingest")
    collection: Optional[str] = Field(None, description="Collection name")


class DocumentResponse(BaseModel):
    """Document information."""

    id: str
    filename: str
    file_type: Optional[str]
    file_path: Optional[str]
    created_at: str


# ===========================================
# Startup Event
# ===========================================


@app.on_event("startup")
async def startup():
    """Initialize databases on startup."""
    initialize_system()


# ===========================================
# Endpoints
# ===========================================


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        info = get_collection_info()
        return {
            "status": "healthy",
            "qdrant_vectors": info["vectors_count"],
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.

    Returns an AI-generated answer based on retrieved context.
    """
    result = rag_query(
        query=request.query,
        model=request.model,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        file_type_filter=request.file_type,
        collection_filter=request.collection,
        include_sources=True,
    )

    return QueryResponse(
        answer=result["answer"],
        model=result["model"],
        contexts_used=result["contexts_used"],
        sources=result.get("sources", []),
    )


@app.post("/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    """
    Retrieve context without calling LLM.

    Useful for debugging or custom LLM integration.
    """
    contexts = retrieve_context(
        query=request.query,
        top_k=request.top_k,
        file_type_filter=request.file_type,
        collection_filter=request.collection,
    )

    augmented_prompt = build_rag_prompt(request.query, contexts)

    return ContextResponse(
        augmented_prompt=augmented_prompt,
        contexts=contexts,
    )


@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    collection: Optional[str] = Query(None, description="Collection name"),
):
    """Upload and ingest a file."""
    # Save to temp file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc_id = ingest_document(tmp_path, collection_name=collection)
        return {
            "status": "success",
            "document_id": str(doc_id),
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/ingest/url")
async def ingest_url(request: IngestURLRequest):
    """Ingest a web page by URL."""
    try:
        doc_id = ingest_document(request.url, collection_name=request.collection)
        return {
            "status": "success",
            "document_id": str(doc_id),
            "url": request.url,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    """List all ingested documents."""
    session = SessionLocal()
    try:
        documents = get_all_documents(session)
        return [
            DocumentResponse(
                id=str(doc.id),
                filename=doc.filename,
                file_type=doc.file_type,
                file_path=doc.file_path,
                created_at=doc.created_at.isoformat(),
            )
            for doc in documents
        ]
    finally:
        session.close()


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors."""
    session = SessionLocal()
    try:
        doc_uuid = uuid.UUID(document_id)
        doc = session.query(Document).filter_by(id=doc_uuid).first()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete from Qdrant
        delete_by_document(doc_uuid)

        # Delete from PostgreSQL (cascades to chunks)
        session.delete(doc)
        session.commit()

        return {"status": "deleted", "document_id": document_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    finally:
        session.close()


# ===========================================
# Run with: uvicorn api.server:app --host 0.0.0.0 --port 8000
# ===========================================
```

---

## Part 7: Create README and Final Files

### Step 7.1: Update README.md

- [ ] Create/update `README.md`:

```markdown
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

See `docker/` folder for docker-compose configuration.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Install Dependencies

```bash
uv sync  # or: pip install -r requirements.txt
```

### 4. Initialize System

```python
from ingestion.pipeline import initialize_system
initialize_system()
```

### 5. Ingest Documents

```python
from ingestion.pipeline import ingest_document
doc_id = ingest_document("path/to/document.pdf")
```

### 6. Query

```python
from retrieval.retriever import rag_query
result = rag_query("What is this document about?")
print(result["answer"])
```

## API Server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Endpoints

- `POST /query` - Query with RAG
- `POST /ingest/file` - Upload document
- `POST /ingest/url` - Ingest web page
- `GET /documents` - List documents
- `DELETE /documents/{id}` - Delete document

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
└── docker/                # NAS deployment
    ├── docker-compose.yml
    └── litellm_config.yaml
```

## License

MIT
```

### Step 7.2: Create Docker Folder for NAS Reference

- [ ] Create `docker/` folder with reference configs:

```bash
mkdir docker
```

- [ ] Create `docker/docker-compose.yml` (copy from Part 1)
- [ ] Create `docker/litellm_config.yaml` (copy from Part 1)
- [ ] Create `docker/.env.example`:

```env
# NAS Docker Environment
OPENAI_API_KEY=sk-proj-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
POSTGRES_PASSWORD=secure-password
LITELLM_MASTER_KEY=sk-litellm-key
```

---

## Part 8: Testing Everything

### Step 8.1: Verify NAS Services

- [ ] All Docker containers running: `docker-compose ps`
- [ ] Qdrant dashboard accessible: `http://NAS-IP:6333/dashboard`
- [ ] PostgreSQL accepting connections
- [ ] LiteLLM health check: `http://NAS-IP:4000/health`
- [ ] OpenWebUI accessible: `http://NAS-IP:3000`

### Step 8.2: Test Python Connection

- [ ] Run in Python/Jupyter:

```python
# Test imports
from config import NAS_IP, DATABASE_URL, LITELLM_BASE_URL
print(f"NAS IP: {NAS_IP}")
print(f"Database: {DATABASE_URL}")
print(f"LiteLLM: {LITELLM_BASE_URL}")
```

### Step 8.3: Initialize System

- [ ] Run initialization:

```python
from ingestion.pipeline import initialize_system
initialize_system()
```

### Step 8.4: Ingest Test Document

- [ ] Create a test file or use existing:

```python
from ingestion.pipeline import ingest_document

# Ingest a PDF
doc_id = ingest_document("path/to/test.pdf")
print(f"Document ID: {doc_id}")

# Or ingest a web page
doc_id = ingest_document("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
```

### Step 8.5: Test Retrieval

- [ ] Query the system:

```python
from retrieval.retriever import rag_query

result = rag_query(
    query="What is RAG?",
    model="gpt-4o-mini",
    top_k=3,
)

print("Answer:", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"  - {source['document']} (relevance: {source['relevance']:.2f})")
```

### Step 8.6: Test API Server

- [ ] Start the server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

- [ ] Test with curl or browser:

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 3}'
```

---

## Troubleshooting

### Connection Errors

- [ ] Verify NAS IP in `.env` is correct
- [ ] Check if Docker containers are running on NAS
- [ ] Ensure ports are not blocked by firewall
- [ ] Test with `ping YOUR-NAS-IP`

### Embedding Errors

- [ ] Verify `OPENAI_API_KEY` is set correctly
- [ ] Check LiteLLM logs: `docker-compose logs litellm`
- [ ] Test LiteLLM directly: `curl http://NAS-IP:4000/health`

### Database Errors

- [ ] PostgreSQL password matches in both `.env` files
- [ ] Qdrant collection exists (run `initialize_system()`)
- [ ] Check PostgreSQL logs: `docker-compose logs postgres`

---

## Next Steps

After completing this tutorial:

1. [ ] Ingest your actual documents
2. [ ] Set up OpenWebUI pipeline for automatic RAG
3. [ ] Create custom bots in OpenWebUI
4. [ ] Experiment with different chunking strategies
5. [ ] Add more document types (Word, Excel, etc.)
6. [ ] Implement hybrid search (vector + keyword)

---

**Tutorial Complete!** Your RAG system should now be fully operational.
