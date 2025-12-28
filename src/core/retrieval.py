"""Vector retrieval service.

This module provides functionality to retrieve relevant documents from
the vector store based on semantic similarity.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchParams,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

from src.config import QdrantConfig, RetrievalConfig, get_settings
from src.core.embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """A candidate document from retrieval."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query: str
    candidates: List[RetrievalCandidate]
    latency_ms: float
    model_version: str


class VectorRetriever:
    """Service for retrieving documents using vector similarity.

    Uses Qdrant as the vector store backend.
    """

    def __init__(
        self,
        qdrant_config: Optional[QdrantConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        embedder: Optional[Embedder] = None,
        client: Optional["QdrantClient"] = None,
    ):
        """Initialize the retriever.

        Args:
            qdrant_config: Qdrant connection configuration
            retrieval_config: Retrieval parameters
            embedder: Embedding service
            client: Pre-initialized Qdrant client (for testing)
        """
        settings = get_settings()
        self.qdrant_config = qdrant_config or settings.qdrant
        self.retrieval_config = retrieval_config or settings.retrieval
        self.embedder = embedder or get_embedder()
        self._client = client
        self._initialized = client is not None

    def _ensure_initialized(self):
        """Lazy initialization of Qdrant client."""
        if not self._initialized:
            if not HAS_QDRANT:
                raise ImportError(
                    "qdrant-client is required. "
                    "Install with: pip install qdrant-client"
                )
            logger.info(f"Connecting to Qdrant: {self.qdrant_config.url}")
            self._client = QdrantClient(url=self.qdrant_config.url)
            self._ensure_collection()
            self._initialized = True

    def _ensure_collection(self):
        """Ensure the collection exists."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.qdrant_config.collection_name for c in collections)

        if not exists:
            logger.info(f"Creating collection: {self.qdrant_config.collection_name}")
            self._client.create_collection(
                collection_name=self.qdrant_config.collection_name,
                vectors_config=VectorParams(
                    size=self.qdrant_config.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            include_embeddings: Include embeddings in results

        Returns:
            RetrievalResult with candidates
        """
        import time
        start_time = time.time()

        self._ensure_initialized()

        top_k = top_k or self.retrieval_config.top_k_candidates

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Build Qdrant filter
        qdrant_filter = self._build_filter(filters) if filters else None

        # Search
        results = self._client.search(
            collection_name=self.qdrant_config.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=include_embeddings,
        )

        # Convert to candidates
        candidates = []
        for result in results:
            if result.score < self.retrieval_config.min_score:
                continue

            candidate = RetrievalCandidate(
                chunk_id=str(result.id),
                document_id=result.payload.get("document_id", ""),
                content=result.payload.get("content", ""),
                score=result.score,
                metadata=result.payload.get("metadata", {}),
                embedding=np.array(result.vector) if result.vector else None,
            )
            candidates.append(candidate)

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            candidates=candidates,
            latency_ms=latency_ms,
            model_version=self.embedder.config.model_name,
        )

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []

        if "document_ids" in filters and filters["document_ids"]:
            for doc_id in filters["document_ids"]:
                conditions.append(
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=doc_id),
                    )
                )

        if "source_types" in filters and filters["source_types"]:
            for source_type in filters["source_types"]:
                conditions.append(
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type),
                    )
                )

        return Filter(should=conditions) if conditions else None

    async def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Index document chunks into the vector store.

        Args:
            chunks: List of chunks with 'id', 'content', 'document_id', 'metadata'
            batch_size: Batch size for indexing

        Returns:
            Number of chunks indexed
        """
        self._ensure_initialized()

        total_indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Generate embeddings
            texts = [chunk["content"] for chunk in batch]
            embeddings = self.embedder.embed_batch(texts)

            # Create points
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                point = PointStruct(
                    id=chunk["id"],
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk["content"],
                        "document_id": chunk["document_id"],
                        "metadata": chunk.get("metadata", {}),
                    },
                )
                points.append(point)

            # Upsert to Qdrant
            self._client.upsert(
                collection_name=self.qdrant_config.collection_name,
                points=points,
            )

            total_indexed += len(points)
            logger.info(f"Indexed {total_indexed}/{len(chunks)} chunks")

        return total_indexed

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            Number of chunks deleted
        """
        self._ensure_initialized()

        # Delete by filter
        self._client.delete(
            collection_name=self.qdrant_config.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

        return 0  # Qdrant doesn't return count


# Singleton instance
_retriever: Optional[VectorRetriever] = None


def get_retriever() -> VectorRetriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever()
    return _retriever
