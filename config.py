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


# ===========================================
# Utility function to print config
# ===========================================
def print_config():
    """Print current configuration (for debugging)."""
    print("=" * 50)
    print("RAG System Configuration")
    print("=" * 50)
    print(f"NAS IP:          {NAS_IP}")
    print(f"Qdrant:          {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"PostgreSQL:      {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    print(f"LiteLLM:         {LITELLM_BASE_URL}")
    print(f"Embedding Model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dims)")
    print(f"Chunk Size:      {CHUNK_SIZE} tokens (overlap: {CHUNK_OVERLAP})")
    print(f"RAG Top-K:       {RAG_TOP_K} (threshold: {RAG_SCORE_THRESHOLD})")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
