"""Document management endpoints."""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import (
    DocumentRequest,
    DocumentResponse,
    DocumentAsyncResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentListItem,
    Pagination,
)
from src.api.dependencies import get_document_service_dep
from src.services.document_service import (
    DocumentService,
    ChunkingConfig,
    ChunkingStrategy,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    request: DocumentRequest,
    service: DocumentService = Depends(get_document_service_dep),
):
    """Ingest a new document into the knowledge base.

    The document will be:
    1. Chunked according to the specified strategy
    2. Embedded using the current embedding model
    3. Indexed in the vector store

    For documents >100KB, processing is asynchronous.
    """
    try:
        # Build chunking config
        chunking = None
        if request.chunking:
            chunking = ChunkingConfig(
                strategy=ChunkingStrategy(request.chunking.strategy),
                chunk_size=request.chunking.chunk_size,
                chunk_overlap=request.chunking.chunk_overlap,
            )

        result = await service.ingest(
            title=request.title,
            content=request.content,
            source_type=request.source_type,
            external_id=request.external_id,
            source_url=request.source_url,
            language=request.language,
            metadata=request.metadata,
            chunking=chunking,
        )

        return DocumentResponse(
            document_id=result.document_id,
            external_id=result.external_id,
            chunks_created=result.chunks_created,
            status=result.status.value,
            estimated_completion=result.estimated_completion,
        )

    except ValueError as e:
        error_msg = str(e)
        if "Duplicate" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "DUPLICATE_DOCUMENT", "message": error_msg},
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_CONTENT", "message": error_msg},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INGEST_ERROR", "message": str(e)},
        )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service_dep),
):
    """Get document metadata and status."""
    document = await service.get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Document not found"},
        )

    return DocumentDetailResponse(
        document_id=document.document_id,
        external_id=document.external_id,
        title=document.title,
        source_type=document.source_type,
        source_url=document.source_url,
        language=document.language,
        metadata=document.metadata,
        status=document.status.value,
        chunks_count=document.chunks_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
        embedding_model=document.embedding_model,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service_dep),
):
    """Remove a document and its chunks from the knowledge base."""
    deleted = await service.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Document not found"},
        )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    source_type: Optional[str] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    search: Optional[str] = None,
    service: DocumentService = Depends(get_document_service_dep),
):
    """List documents with filtering and pagination."""
    result = await service.list_documents(
        page=page,
        per_page=per_page,
        source_type=source_type,
        created_after=created_after,
        created_before=created_before,
        search=search,
    )

    return DocumentListResponse(
        documents=[
            DocumentListItem(
                document_id=d["document_id"],
                external_id=d.get("external_id"),
                title=d["title"],
                source_type=d["source_type"],
                status=d["status"],
                chunks_count=d["chunks_count"],
                created_at=d["created_at"],
                updated_at=d["updated_at"],
            )
            for d in result["documents"]
        ],
        pagination=Pagination(**result["pagination"]),
    )
