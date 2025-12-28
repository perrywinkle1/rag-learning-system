"""API module for RAG Learning System.

This module provides the FastAPI application and endpoints.
"""

from src.api.main import app, create_app

__all__ = ["app", "create_app"]
