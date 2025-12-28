"""FastAPI application for RAG Learning System."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.api.routers import query_router, feedback_router, documents_router, admin_router
from src.api.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Learning System...")
    settings = get_settings()
    logger.info(f"Debug mode: {settings.debug}")
    yield
    # Shutdown
    logger.info("Shutting down RAG Learning System...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="RAG Learning System",
        description="A self-improving Retrieval-Augmented Generation system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else "An unexpected error occurred",
            },
        )

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Check system health."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            services={
                "api": True,
                "database": True,  # Would check actual connection
                "vector_store": True,
                "cache": True,
            },
        )

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "RAG Learning System",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    # Include routers
    api_prefix = settings.api_prefix
    app.include_router(query_router, prefix=api_prefix)
    app.include_router(feedback_router, prefix=api_prefix)
    app.include_router(documents_router, prefix=api_prefix)
    app.include_router(admin_router, prefix=api_prefix)

    return app


# Create app instance
app = create_app()
