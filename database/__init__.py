# database/__init__.py
"""Database connections and models."""
from database.postgres import (
    Document,
    Chunk,
    Collection,
    init_database,
    get_session,
    SessionLocal,
    get_document_by_hash,
    get_all_documents,
    get_document_chunks,
)
from database.qdrant import (
    init_collection,
    store_embeddings,
    search_similar,
    delete_by_document,
    delete_collection,
    get_collection_info,
)

__all__ = [
    # PostgreSQL
    "Document",
    "Chunk",
    "Collection",
    "init_database",
    "get_session",
    "SessionLocal",
    "get_document_by_hash",
    "get_all_documents",
    "get_document_chunks",
    # Qdrant
    "init_collection",
    "store_embeddings",
    "search_similar",
    "delete_by_document",
    "delete_collection",
    "get_collection_info",
]
