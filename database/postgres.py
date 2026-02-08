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
    """Get all documents ordered by creation date."""
    return session.query(Document).order_by(Document.created_at.desc()).all()


def get_document_chunks(session: Session, document_id: uuid.UUID) -> list[Chunk]:
    """Get all chunks for a document ordered by index."""
    return (
        session.query(Chunk)
        .filter_by(document_id=document_id)
        .order_by(Chunk.chunk_index)
        .all()
    )


def get_document_by_id(session: Session, document_id: uuid.UUID) -> Document | None:
    """Get a document by its ID."""
    return session.query(Document).filter_by(id=document_id).first()


def delete_document(session: Session, document_id: uuid.UUID) -> bool:
    """Delete a document and all its chunks."""
    doc = get_document_by_id(session, document_id)
    if doc:
        session.delete(doc)
        session.commit()
        return True
    return False
