"""High-level retrieval service.

This module provides a high-level interface for the complete
retrieval pipeline: query processing -> retrieval -> reranking -> generation.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.config import get_settings
from src.core.query_processor import QueryProcessor, ProcessedQuery, get_query_processor
from src.core.retrieval import VectorRetriever, RetrievalResult, get_retriever
from src.core.reranker import Reranker, RankedCandidate, get_reranker
from src.core.generator import ResponseGenerator, GeneratedResponse, get_generator
from src.services.cache import CacheService, get_cache

logger = logging.getLogger(__name__)


@dataclass
class QueryRequest:
    """A query request from the user."""
    query: str
    session_id: Optional[str] = None
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    """A source document in the response."""
    chunk_id: str
    document_id: str
    document_title: str
    content: str
    rank: int
    score: float
    metadata: Dict[str, Any]


@dataclass
class QueryResponse:
    """Response to a query."""
    response_id: str
    query: str
    answer: str
    sources: List[Source]
    confidence: float
    model_info: Dict[str, str]
    latency_ms: float
    session_id: Optional[str] = None


class RetrievalService:
    """High-level retrieval service.

    Orchestrates the complete query pipeline:
    1. Query preprocessing
    2. Vector retrieval
    3. Re-ranking
    4. Response generation
    5. Caching
    """

    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        retriever: Optional[VectorRetriever] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[ResponseGenerator] = None,
        cache: Optional[CacheService] = None,
    ):
        """Initialize the retrieval service.

        Args:
            query_processor: Query preprocessing service
            retriever: Vector retrieval service
            reranker: Re-ranking service
            generator: Response generation service
            cache: Caching service
        """
        self.query_processor = query_processor or get_query_processor()
        self.retriever = retriever or get_retriever()
        self.reranker = reranker or get_reranker()
        self.generator = generator or get_generator()
        self.cache = cache or get_cache()

        self.settings = get_settings()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Process a query and return a response.

        Args:
            request: Query request

        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        response_id = str(uuid.uuid4())

        # Check cache first
        cached = await self.cache.get_query_result(
            request.query,
            request.filters,
        )
        if cached:
            logger.info(f"Cache hit for query: {request.query[:50]}...")
            return QueryResponse(**cached)

        # Process query
        processed = self.query_processor.process(request.query)

        # Retrieve candidates
        retrieval_result = await self.retriever.retrieve(
            query=processed.expanded_query,
            top_k=self.settings.retrieval.top_k_candidates,
            filters=request.filters,
        )

        # Re-rank candidates
        reranked = await self.reranker.rerank(
            query=request.query,
            candidates=retrieval_result.candidates,
            top_k=request.top_k,
        )

        # Generate response
        generated = await self.generator.generate(
            query=request.query,
            candidates=reranked,
            response_id=response_id,
        )

        # Build response
        sources = [
            Source(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.metadata.get("title", "Unknown"),
                content=c.content,
                rank=c.rank,
                score=c.final_score,
                metadata=c.metadata,
            )
            for c in reranked
        ]

        latency_ms = (time.time() - start_time) * 1000

        response = QueryResponse(
            response_id=response_id,
            query=request.query,
            answer=generated.answer,
            sources=sources,
            confidence=generated.confidence,
            model_info={
                "retrieval_model": self.settings.embedding.model_name,
                "generation_model": self.settings.generation.model,
                "reranker_model": self.settings.reranker.model_name,
            },
            latency_ms=latency_ms,
            session_id=request.session_id,
        )

        # Cache the response
        await self.cache.set_query_result(
            request.query,
            {
                "response_id": response.response_id,
                "query": response.query,
                "answer": response.answer,
                "sources": [
                    {
                        "chunk_id": s.chunk_id,
                        "document_id": s.document_id,
                        "document_title": s.document_title,
                        "content": s.content,
                        "rank": s.rank,
                        "score": s.score,
                        "metadata": s.metadata,
                    }
                    for s in response.sources
                ],
                "confidence": response.confidence,
                "model_info": response.model_info,
                "latency_ms": response.latency_ms,
                "session_id": response.session_id,
            },
            request.filters,
        )

        # Also cache the response by ID for later retrieval
        await self.cache.set_response(
            response_id,
            {
                "response_id": response_id,
                "query": request.query,
                "answer": generated.answer,
                "sources": [s.__dict__ for s in sources],
                "confidence": generated.confidence,
            },
        )

        logger.info(
            f"Query processed: {request.query[:50]}... "
            f"({len(sources)} sources, {latency_ms:.0f}ms)"
        )

        return response

    async def get_response(self, response_id: str) -> Optional[Dict]:
        """Get a previous response by ID.

        Args:
            response_id: Response identifier

        Returns:
            Cached response or None
        """
        return await self.cache.get_response(response_id)

    async def retrieve_only(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[RankedCandidate]:
        """Retrieve and rerank without generating a response.

        Useful for getting just the sources without LLM generation.

        Args:
            query: Search query
            top_k: Number of results
            filters: Query filters

        Returns:
            List of ranked candidates
        """
        # Process query
        processed = self.query_processor.process(query)

        # Retrieve
        retrieval_result = await self.retriever.retrieve(
            query=processed.expanded_query,
            top_k=self.settings.retrieval.top_k_candidates,
            filters=filters,
        )

        # Rerank
        reranked = await self.reranker.rerank(
            query=query,
            candidates=retrieval_result.candidates,
            top_k=top_k,
        )

        return reranked


# Singleton instance
_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Get the global retrieval service instance."""
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service
