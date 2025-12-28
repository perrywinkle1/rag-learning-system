"""API routers for RAG Learning System."""

from src.api.routers.query import router as query_router
from src.api.routers.feedback import router as feedback_router
from src.api.routers.documents import router as documents_router
from src.api.routers.admin import router as admin_router

__all__ = [
    "query_router",
    "feedback_router",
    "documents_router",
    "admin_router",
]
