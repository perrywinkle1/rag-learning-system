"""Core RAG components.

This module provides the core functionality for:
- Text embedding
- Vector retrieval
- Re-ranking
- Query processing
- Response generation
"""

from src.core.embedder import Embedder, get_embedder
from src.core.retrieval import VectorRetriever, RetrievalCandidate, RetrievalResult, get_retriever
from src.core.reranker import Reranker, HybridReranker, RankedCandidate, get_reranker
from src.core.query_processor import QueryProcessor, ProcessedQuery, QueryRewriter, get_query_processor
from src.core.generator import ResponseGenerator, GeneratedResponse, Citation, StreamChunk, get_generator

__all__ = [
    # Embedder
    "Embedder",
    "get_embedder",
    # Retrieval
    "VectorRetriever",
    "RetrievalCandidate",
    "RetrievalResult",
    "get_retriever",
    # Reranker
    "Reranker",
    "HybridReranker",
    "RankedCandidate",
    "get_reranker",
    # Query processor
    "QueryProcessor",
    "ProcessedQuery",
    "QueryRewriter",
    "get_query_processor",
    # Generator
    "ResponseGenerator",
    "GeneratedResponse",
    "Citation",
    "StreamChunk",
    "get_generator",
]
