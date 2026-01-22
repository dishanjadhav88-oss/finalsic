import logging
from typing import List
from pathlib import Path

from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP


# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data-loader")


# -------------------------------------------------
# PDF Text Extraction
# -------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    """
    Extract raw text from a PDF file.
    """
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages_text: List[str] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        except Exception:
            logger.warning(f"Failed to extract text from page {i}")

    return "\n".join(pages_text)


# -------------------------------------------------
# Text Chunking (Safe Overlap)
# -------------------------------------------------
def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks.
    """
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks


# -------------------------------------------------
# PDF → Chunks
# -------------------------------------------------
def load_and_chunk_pdf(path: str) -> List[str]:
    """
    Load PDF and split into text chunks.
    """
    logger.info(f"Loading PDF: {path}")

    text = extract_text_from_pdf(path)

    if not text.strip():
        logger.warning("No text extracted from PDF")
        return []

    chunks = chunk_text(text)

    logger.info("Extracted %d chunks", len(chunks))
    return chunks


def embed_texts():
    return None