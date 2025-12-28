"""Service layer for RAG Learning System.

This module provides high-level services for:
- Query orchestration (retrieval + generation)
- Feedback collection
- Document management
- Caching
"""

from src.services.cache import CacheService, get_cache
from src.services.retrieval_service import (
    RetrievalService,
    QueryRequest,
    QueryResponse,
    Source,
    get_retrieval_service,
)
from src.services.feedback_service import (
    FeedbackService,
    FeedbackEvent,
    FeedbackType,
    FeedbackReason,
    FeedbackResult,
    get_feedback_service,
)
from src.services.document_service import (
    DocumentService,
    Document,
    DocumentChunk,
    DocumentStatus,
    ChunkingStrategy,
    ChunkingConfig,
    IngestResult,
    get_document_service,
)

__all__ = [
    # Cache
    "CacheService",
    "get_cache",
    # Retrieval
    "RetrievalService",
    "QueryRequest",
    "QueryResponse",
    "Source",
    "get_retrieval_service",
    # Feedback
    "FeedbackService",
    "FeedbackEvent",
    "FeedbackType",
    "FeedbackReason",
    "FeedbackResult",
    "get_feedback_service",
    # Documents
    "DocumentService",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "ChunkingStrategy",
    "ChunkingConfig",
    "IngestResult",
    "get_document_service",
]
