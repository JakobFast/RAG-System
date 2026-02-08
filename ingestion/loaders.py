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
