"""Feedback endpoints."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    BatchFeedbackRequest,
    BatchFeedbackResponse,
)
from src.api.dependencies import get_feedback_service_dep
from src.services.feedback_service import (
    FeedbackService,
    FeedbackType,
    FeedbackReason as FeedbackReasonEnum,
)

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(
    request: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service_dep),
):
    """Submit feedback on a response.

    Accepts various feedback types:
    - thumbs_up / thumbs_down: Simple binary feedback
    - rating: 1-5 star rating
    - click: User clicked on a source
    - dwell: Time spent reading (in ms)
    - copy: User copied content
    - share: User shared the response

    Feedback is used to improve the retrieval model.
    """
    try:
        # Map reason code
        reason_code = None
        if request.reason and request.reason.code:
            reason_code = FeedbackReasonEnum(request.reason.code.value)

        result = await service.submit_feedback(
            response_id=request.response_id,
            feedback_type=FeedbackType(request.type.value),
            value=request.value,
            target_chunk_id=request.target_chunk_id,
            reason_code=reason_code,
            reason_text=request.reason.text if request.reason else None,
            client_timestamp=request.client_timestamp,
        )

        if not result.accepted:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "INVALID_FEEDBACK", "message": result.error},
            )

        return FeedbackResponse(
            feedback_id=result.feedback_id,
            accepted=result.accepted,
            message=result.message,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_VALUE", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "FEEDBACK_ERROR", "message": str(e)},
        )


@router.post("/batch", response_model=BatchFeedbackResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_batch_feedback(
    request: BatchFeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service_dep),
):
    """Submit multiple feedback events in one request.

    Useful for sending batched behavioral signals (clicks, dwell times, etc.)
    collected on the client side.
    """
    try:
        events = [
            {
                "response_id": e.response_id,
                "type": e.type.value,
                "value": e.value,
                "target_chunk_id": e.target_chunk_id,
                "client_timestamp": e.client_timestamp.isoformat() if e.client_timestamp else None,
            }
            for e in request.events
        ]

        result = await service.submit_batch(events)

        return BatchFeedbackResponse(
            accepted=result["accepted"],
            rejected=result["rejected"],
            errors=result["errors"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "BATCH_ERROR", "message": str(e)},
        )
