import os
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()


# =================================================
# Gemini (Google AI)
# =================================================
from typing import Optional

GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in environment variables")

# Recommended Gemini models (2025)
GEMINI_LLM_MODEL = "gemini-2.5-flash"
GEMINI_EMBED_MODEL = "text-embedding-004"
EMBED_DIM = 768


# =================================================
# Qdrant Vector Database
# =================================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "legal_docs"


# =================================================
# PDF / Chunking
# =================================================
BASE_DATA_DIR = Path("shared_data")
UPLOAD_DIR = BASE_DATA_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# =================================================
# Legal Document Types (Classifier Targets)
# =================================================
LEGAL_DOCUMENT_TYPES = {
    "FIR": "First Information Report",
    "COURT_ORDER": "Court judgment or order",
    "AFFIDAVIT": "Sworn affidavit",
    "PETITION": "Legal petition",
    "AGREEMENT": "Legal agreement",
    "GOVERNMENT_NOTICE": "Official notice",
    "OTHER": "Not a legal document",
}