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
