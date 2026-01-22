import logging
from typing import List, Optional

import google.generativeai as genai

from config import (
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
    EMBED_DIM,
)

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-embedder")


# =================================================
# Gemini Embedder
# =================================================
class GeminiEmbedder:
    """
    Handles document and query embeddings using Gemini
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=GEMINI_API_KEY)
        self.model = GEMINI_EMBED_MODEL

        logger.info("GeminiEmbedder initialized (model=%s)", self.model)

    # -------------------------------------------------
    # Document embeddings
    # -------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed document chunks for vector storage.
        """
        if not texts:
            logger.warning("No texts provided for document embedding")
            return []

        try:
            logger.info("Embedding %d document chunks", len(texts))

            response = genai.embed_content(
                model=self.model,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT",
            )

            embeddings = response["embedding"]

            if len(embeddings) != len(texts):
                raise RuntimeError("Embedding count mismatch")

            # Safety check
            for vec in embeddings:
                if len(vec) != EMBED_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(vec)}"
                    )

            return embeddings

        except Exception as e:
            logger.exception("Gemini document embedding failed")
            raise RuntimeError("Gemini document embedding error") from e

    # -------------------------------------------------
    # Query embedding
    # -------------------------------------------------
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query for similarity search.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            logger.info("Embedding query")

            response = genai.embed_content(
                model=self.model,
                content=query,
                task_type="RETRIEVAL_QUERY",
            )

            vector = response["embedding"]

            if len(vector) != EMBED_DIM:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {EMBED_DIM}, got {len(vector)}"
                )

            return vector

        except Exception as e:
            logger.exception("Gemini query embedding failed")
            raise RuntimeError("Gemini query embedding error") from e


# =================================================
# Singleton accessor (worker-safe)
# =================================================

from typing import Optional
_embedder: Optional[GeminiEmbedder] = None

def get_embedder() -> GeminiEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = GeminiEmbedder()
    return _embedder