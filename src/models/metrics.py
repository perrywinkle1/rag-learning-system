"""Metrics and monitoring models."""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, ForeignKey, Index, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class RetrievalMetric(Base, TimestampMixin):
    """Retrieval quality metrics per query."""

    __tablename__ = "retrieval_metrics"

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
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    mrr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ndcg_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ndcg_at_10: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_at_10: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_at_10: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    click_through_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    time_to_first_click: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_metrics_response", "response_id"),
        Index("idx_metrics_model", "model_version"),
        Index("idx_metrics_calculated", "calculated_at"),
    )


class SystemMetric(Base, TimestampMixin):
    """Aggregated system-level metrics."""

    __tablename__ = "system_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    experiment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ab_experiments.id"),
        nullable=True
    )
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("idx_sysmetrics_name", "metric_name"),
        Index("idx_sysmetrics_model", "model_version"),
        Index("idx_sysmetrics_period", "period_start", "period_end"),
    )
