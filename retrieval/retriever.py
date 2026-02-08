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
                doc = session.query(Document).filter_by(id=chunk.document_id).first()
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
