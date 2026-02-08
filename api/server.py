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

Run with: uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
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
            "qdrant_vectors": info.get("vectors_count", 0),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.

    Returns an AI-generated answer based on retrieved context.
    """
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    """
    Retrieve context without calling LLM.

    Useful for debugging or custom LLM integration.
    """
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()
