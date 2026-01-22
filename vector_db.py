import logging
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import QDRANT_URL, QDRANT_COLLECTION, EMBED_DIM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qdrant-storage")


class QdrantStorage:
    """
    Wrapper around QdrantClient for vector storage and retrieval.
    """

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection: str = QDRANT_COLLECTION,
        dim: int = EMBED_DIM,
    ):
        try:
            self.client = QdrantClient(url=url, timeout=30)
            self.collection = collection
            self.dim = dim

            # Create collection if it doesn't exist
            if not self.client.collection_exists(collection_name=self.collection):
                logger.info(f"Creating Qdrant collection: {self.collection}")
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.dim,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection}")

        except Exception as e:
            logger.exception("Failed to initialize Qdrant client")
            raise RuntimeError("Qdrant initialization failed") from e

    # -------------------------------------------------
    # Upsert
    # -------------------------------------------------
    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict],
    ) -> bool:
        if not (ids and vectors and payloads):
            logger.warning("Upsert skipped: empty ids, vectors, or payloads")
            return False

        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must have equal length")

        try:
            points = [
                PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload=payloads[i],
                )
                for i in range(len(ids))
            ]

            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )

            logger.info(f"Upserted {len(points)} vectors")
            return True

        except Exception as e:
            logger.exception("Vector upsert failed")
            return False

    # -------------------------------------------------
    # Search
    # -------------------------------------------------
    def search(self, query_vector: List[float], top_k: int = 10) -> Dict[str, List[str]]:
        if not query_vector:
            return {"contexts": [], "sources": []}

        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

            contexts: List[str] = []
            sources: set[str] = set()

            for hit in results:
                payload = hit.payload or {}

                text = payload.get("text")
                source = payload.get("source")
                doc_type = payload.get("doc_type", "UNKNOWN")

                if text:
                    contexts.append(f"[{doc_type}] {text}")
                if source:
                    sources.add(source)

            logger.info(
                "Search returned %d contexts from %d sources",
                len(contexts),
                len(sources),
            )

            return {
                "contexts": contexts,
                "sources": list(sources),
            }

        except Exception as e:
            logger.exception("Vector search failed")
            return {"contexts": [], "sources": []}

    # -------------------------------------------------
    # Delete by Source
    # -------------------------------------------------
    def delete_by_source(self, source_id: str) -> bool:
        if not source_id:
            return False

        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_id),
                        )
                    ]
                ),
            )

            logger.info(f"Deleted vectors for source: {source_id}")
            return True

        except Exception as e:
            logger.exception("Delete by source failed")
            return False