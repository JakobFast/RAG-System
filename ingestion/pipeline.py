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
        code_types = ["py", "js", "ts", "java", "c", "cpp", "go", "rs", "rb", "php"]
        if use_code_chunking or file_type in code_types:
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
