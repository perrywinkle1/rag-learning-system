"""Query and response models."""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Text, Integer, Float, ForeignKey, Index, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin


class Query(Base, TimestampMixin):
    """User query storage."""

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    responses: Mapped[List["QueryResponse"]] = relationship(
        "QueryResponse",
        back_populates="query",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_queries_session", "session_id"),
        Index("idx_queries_user", "user_id"),
        Index("idx_queries_hash", "query_hash"),
    )


class QueryResponse(Base, TimestampMixin):
    """Generated response to a query."""

    __tablename__ = "query_responses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=False
    )
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    generation_model: Mapped[str] = mapped_column(String(100), nullable=False)
    experiment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ab_experiments.id"),
        nullable=True
    )
    is_control: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    # Relationships
    query: Mapped["Query"] = relationship("Query", back_populates="responses")
    retrieved_chunks: Mapped[List["RetrievedChunk"]] = relationship(
        "RetrievedChunk",
        back_populates="response",
        cascade="all, delete-orphan"
    )
    feedback_events: Mapped[List["FeedbackEvent"]] = relationship(
        "FeedbackEvent",
        back_populates="response",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_responses_query", "query_id"),
        Index("idx_responses_model", "retrieval_model_version"),
        Index("idx_responses_experiment", "experiment_id"),
    )


class RetrievedChunk(Base, TimestampMixin):
    """Chunk retrieved for a response."""

    __tablename__ = "retrieved_chunks"

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
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    rerank_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    was_used_in_answer: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    response: Mapped["QueryResponse"] = relationship(
        "QueryResponse",
        back_populates="retrieved_chunks"
    )

    __table_args__ = (
        Index("idx_retrieved_response", "response_id"),
        Index("idx_retrieved_chunk", "chunk_id"),
    )


# Import for relationship resolution
from src.models.feedback import FeedbackEvent
