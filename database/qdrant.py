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
    FilterSelector,
    PointIdsList,
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
    try:
        info = client.get_collection(collection_name=QDRANT_COLLECTION)
        return {
            "name": QDRANT_COLLECTION,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {
            "name": QDRANT_COLLECTION,
            "error": str(e),
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
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id)),
                    )
                ]
            )
        ),
    )


def delete_points(point_ids: list[str]) -> None:
    """Delete specific points by ID."""
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=PointIdsList(points=point_ids),
    )
