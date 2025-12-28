"""FastAPI dependency injection."""

from typing import Optional
from functools import lru_cache

from src.config import Settings, get_settings
from src.services.retrieval_service import RetrievalService, get_retrieval_service
from src.services.feedback_service import FeedbackService, get_feedback_service
from src.services.document_service import DocumentService, get_document_service
from src.services.cache import CacheService, get_cache


def get_settings_dep() -> Settings:
    """Get application settings."""
    return get_settings()


def get_retrieval_service_dep() -> RetrievalService:
    """Get retrieval service dependency."""
    return get_retrieval_service()


def get_feedback_service_dep() -> FeedbackService:
    """Get feedback service dependency."""
    return get_feedback_service()


def get_document_service_dep() -> DocumentService:
    """Get document service dependency."""
    return get_document_service()


def get_cache_dep() -> CacheService:
    """Get cache service dependency."""
    return get_cache()
