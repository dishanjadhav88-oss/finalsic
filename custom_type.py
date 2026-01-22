from typing import List, Optional
from pydantic import BaseModel, Field, confloat


# ==================================================
# Chunk + Source (Ingest Pipeline State)
# ==================================================
class RAGChunkAndSrc(BaseModel):
    chunks: List[str] = Field(
        default_factory=list,
        description="Text chunks extracted from the PDF"
    )
    source_id: str = Field(
        ...,
        description="Original PDF filename or unique source identifier"
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="Predicted legal document type"
    )
    confidence: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None,
        description="Classifier confidence score (0.0 - 1.0)"
    )


# ==================================================
# Upsert Result (Vector DB Insert)
# ==================================================
class RAGUpsertResult(BaseModel):
    ingested: int = Field(
        ...,
        ge=0,
        description="Number of chunks successfully ingested"
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Source document ID"
    )
    doc_type: Optional[str] = Field(
        default="UNKNOWN",
        description="Detected legal document type"
    )


# ==================================================
# Vector Search Result
# ==================================================
class RAGSearchResult(BaseModel):
    contexts: List[str] = Field(
        default_factory=list,
        description="Retrieved context chunks"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source document identifiers"
    )


# ==================================================
# Final Query Result (API Output)
# ==================================================
class RAGQueryResult(BaseModel):
    answer: str = Field(
        ...,
        description="LLM-generated answer based on retrieved context"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Documents used to generate the answer"
    )
    num_contexts: int = Field(
        ...,
        ge=0,
        description="Number of context chunks used"
    )
    doc_type: Optional[str] = Field(
        default="UNKNOWN",
        description="Primary legal document type used"
    )
    confidence: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None,
        description="Confidence of document classification"
    )