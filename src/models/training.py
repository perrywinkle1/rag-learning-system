"""Training and model management models."""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, ForeignKey, Index, Boolean, Enum as SQLEnum, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from src.models.base import Base, TimestampMixin


class TrainingPair(Base, TimestampMixin):
    """Contrastive training pair from feedback."""

    __tablename__ = "training_pairs"

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
    positive_chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False
    )
    negative_chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    source_feedback_ids: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    used_in_training: Mapped[bool] = mapped_column(default=False)
    training_batch_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        Index("idx_pairs_query", "query_id"),
        Index("idx_pairs_positive", "positive_chunk_id"),
        Index("idx_pairs_negative", "negative_chunk_id"),
        Index("idx_pairs_confidence", "confidence_score"),
        Index("idx_pairs_unused", "used_in_training"),
    )


class ModelStatus(enum.Enum):
    """Model deployment status."""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGED = "staged"
    CANARY = "canary"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class ModelVersion(Base, TimestampMixin):
    """Model version registry."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # embedding, reranker
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ModelStatus] = mapped_column(
        SQLEnum(ModelStatus),
        default=ModelStatus.TRAINING,
        nullable=False
    )
    artifact_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    training_config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    training_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    validation_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    training_pairs_count: Mapped[int] = mapped_column(Integer, default=0)
    parent_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id"),
        nullable=True
    )
    deployed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    retired_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    __table_args__ = (
        Index("idx_models_status", "status"),
        Index("idx_models_type", "model_type"),
        Index("idx_models_version", "version"),
    )


class ExperimentStatus(enum.Enum):
    """A/B experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABExperiment(Base, TimestampMixin):
    """A/B testing experiment configuration."""

    __tablename__ = "ab_experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[ExperimentStatus] = mapped_column(
        SQLEnum(ExperimentStatus),
        default=ExperimentStatus.DRAFT,
        nullable=False
    )
    control_model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id"),
        nullable=False
    )
    treatment_model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id"),
        nullable=False
    )
    traffic_percentage: Mapped[float] = mapped_column(Float, default=0.1)  # 10% to treatment
    min_sample_size: Mapped[int] = mapped_column(Integer, default=1000)
    target_metric: Mapped[str] = mapped_column(String(50), default="mrr_at_10")
    min_improvement: Mapped[float] = mapped_column(Float, default=0.05)  # 5% improvement
    start_time: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    results: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("idx_experiments_status", "status"),
        Index("idx_experiments_control", "control_model_id"),
        Index("idx_experiments_treatment", "treatment_model_id"),
    )
