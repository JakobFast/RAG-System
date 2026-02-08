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
