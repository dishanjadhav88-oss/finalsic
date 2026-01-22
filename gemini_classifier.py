import logging
from typing import Tuple, Optional

import google.generativeai as genai

from config import (
    GEMINI_API_KEY,
    GEMINI_LLM_MODEL,
    LEGAL_DOCUMENT_TYPES,
)

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-classifier")

# -------------------------------------------------
# Prompt Template
# -------------------------------------------------
CLASSIFICATION_PROMPT = """
You are a legal document classifier.

Possible categories:
{categories}

Analyze the document below and respond in EXACT format:

DOCUMENT_TYPE | CONFIDENCE

Where:
- DOCUMENT_TYPE is one of the categories
- CONFIDENCE is a number between 0.0 and 1.0

Document:
{content}
"""


# =================================================
# Document Classifier
# =================================================
class DocumentClassifier:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not found")

        genai.configure(api_key=GEMINI_API_KEY)

        self.model = genai.GenerativeModel(GEMINI_LLM_MODEL)
        self.categories = ", ".join(LEGAL_DOCUMENT_TYPES.keys())

        logger.info(
            "DocumentClassifier initialized (model=%s)",
            GEMINI_LLM_MODEL,
        )

    def classify_document(
        self,
        text: str,
        max_chars: int = 2000,
    ) -> Tuple[str, float]:
        """
        Classify a document and return (doc_type, confidence).

        confidence ∈ [0.0, 1.0]
        """

        if not text or not text.strip():
            logger.warning("Empty document text")
            return "OTHER", 0.0

        try:
            truncated_text = text[:max_chars]

            prompt = CLASSIFICATION_PROMPT.format(
                categories=self.categories,
                content=truncated_text,
            )

            response = self.model.generate_content(prompt)

            if not response or not getattr(response, "text", None):
                logger.warning("Empty response from Gemini")
                return "OTHER", 0.0

            raw = response.text.strip()

            if "|" not in raw:
                logger.warning("Unexpected classifier output: %s", raw)
                return "OTHER", 0.0

            doc_type_raw, conf_raw = raw.split("|", 1)
            doc_type = doc_type_raw.strip().upper()

            try:
                confidence = float(conf_raw.strip())
            except ValueError:
                confidence = 0.0

            # Clamp confidence
            confidence = max(0.0, min(confidence, 1.0))

            if doc_type not in LEGAL_DOCUMENT_TYPES:
                logger.warning("Unknown document type predicted: %s", doc_type)
                return "OTHER", 0.0

            logger.info(
                "Document classified as %s (confidence=%.2f)",
                doc_type,
                confidence,
            )

            return doc_type, confidence

        except Exception:
            logger.exception("Document classification failed")
            return "OTHER", 0.0


# =================================================
# Singleton Accessor (Worker-safe)
# =================================================
from typing import Optional
_classifier: Optional[DocumentClassifier] = None


def get_classifier() -> DocumentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifier()
    return _classifier