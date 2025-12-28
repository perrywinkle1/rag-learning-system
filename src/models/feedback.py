"""Feedback event models."""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, ForeignKey, Index, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from src.models.base import Base, TimestampMixin


class FeedbackType(enum.Enum):
    """Types of feedback events."""
    # Explicit feedback
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"

    # Implicit feedback
    CLICK = "click"
    DWELL = "dwell"
    SCROLL_DEPTH = "scroll_depth"
    COPY = "copy"
    SHARE = "share"
    FOLLOW_UP = "follow_up"
    REFORMULATION = "reformulation"
    ABANDONMENT = "abandonment"


class FeedbackEvent(Base, TimestampMixin):
    """User feedback event storage."""

    __tablename__ = "feedback_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    response_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_responses.id", ondelete="CASCADE"),
        nullable=False
    )
    event_type: Mapped[FeedbackType] = mapped_column(
        SQLEnum(FeedbackType),
        nullable=False
    )
    event_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True
    )
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    processed: Mapped[bool] = mapped_column(default=False)
    processed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    # Relationships
    response: Mapped["QueryResponse"] = relationship(
        "QueryResponse",
        back_populates="feedback_events"
    )

    __table_args__ = (
        Index("idx_feedback_response", "response_id"),
        Index("idx_feedback_type", "event_type"),
        Index("idx_feedback_processed", "processed"),
        Index("idx_feedback_session", "session_id"),
        Index("idx_feedback_created", "created_at"),
    )


# Import for relationship resolution
from src.models.query import QueryResponse
