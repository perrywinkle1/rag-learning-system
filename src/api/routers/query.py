"""Query endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import json

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    QueryHistoryResponse,
    SourceResponse,
    ModelInfo,
)
from src.api.dependencies import get_retrieval_service_dep
from src.services.retrieval_service import (
    RetrievalService,
    QueryRequest as ServiceQueryRequest,
)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: RetrievalService = Depends(get_retrieval_service_dep),
):
    """Submit a query and receive a generated response with sources.

    This endpoint:
    1. Preprocesses the query
    2. Retrieves relevant documents
    3. Re-ranks results
    4. Generates a response using LLM

    Returns the answer with sources and confidence score.
    """
    try:
        # Convert to service request
        service_request = ServiceQueryRequest(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            filters=request.filters.model_dump() if request.filters else None,
            options=request.options.model_dump() if request.options else {},
        )

        # Check for streaming
        if request.options and request.options.stream:
            return await _stream_query(service_request, service)

        # Execute query
        response = await service.query(service_request)

        return QueryResponse(
            response_id=response.response_id,
            answer=response.answer,
            sources=[
                SourceResponse(
                    chunk_id=s.chunk_id,
                    document_id=s.document_id,
                    document_title=s.document_title,
                    content=s.content,
                    rank=s.rank,
                    score=s.score,
                    metadata=s.metadata,
                )
                for s in response.sources
            ],
            confidence=response.confidence,
            model_info=ModelInfo(**response.model_info),
            latency_ms=response.latency_ms,
            session_id=response.session_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_QUERY", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "RETRIEVAL_ERROR", "message": str(e)},
        )


async def _stream_query(request: ServiceQueryRequest, service: RetrievalService):
    """Stream query response using Server-Sent Events."""

    async def event_generator():
        try:
            # Get sources first
            sources = await service.retrieve_only(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
            )

            # Send sources
            for i, source in enumerate(sources):
                yield f"event: source\n"
                yield f"data: {json.dumps({'chunk_id': source.chunk_id, 'content': source.content[:500], 'rank': source.rank, 'score': source.final_score})}\n\n"

            # Send answer start
            yield f"event: answer_start\n"
            yield f"data: {json.dumps({'response_id': 'stream-response'})}\n\n"

            # For now, return a placeholder (would implement actual streaming)
            yield f"event: answer_chunk\n"
            yield f"data: {json.dumps({'text': 'Streaming response generation coming soon...'})}\n\n"

            # Send answer end
            yield f"event: answer_end\n"
            yield f"data: {json.dumps({'confidence': 0.8, 'latency_ms': 0})}\n\n"

        except Exception as e:
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/{response_id}", response_model=QueryHistoryResponse)
async def get_response(
    response_id: str,
    service: RetrievalService = Depends(get_retrieval_service_dep),
):
    """Retrieve a previous response by ID.

    Returns the original query, answer, sources, and feedback summary.
    """
    response = await service.get_response(response_id)

    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Response not found"},
        )

    return QueryHistoryResponse(
        response_id=response["response_id"],
        query=response.get("query", ""),
        answer=response["answer"],
        sources=[
            SourceResponse(**s) for s in response.get("sources", [])
        ],
        confidence=response["confidence"],
        created_at=response.get("created_at", "2024-01-01T00:00:00Z"),
        feedback_summary=response.get("feedback_summary"),
    )
