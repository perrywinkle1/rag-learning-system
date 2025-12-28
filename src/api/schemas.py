"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ==================== Query Schemas ====================

class QueryFilters(BaseModel):
    """Filters for query requests."""
    document_ids: Optional[List[str]] = None
    source_types: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryOptions(BaseModel):
    """Options for query requests."""
    include_scores: bool = False
    include_embeddings: bool = False
    stream: bool = False


class QueryRequest(BaseModel):
    """Request schema for POST /query."""
    query: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[QueryFilters] = None
    options: Optional[QueryOptions] = None


class SourceResponse(BaseModel):
    """A source document in the response."""
    chunk_id: str
    document_id: str
    document_title: str
    content: str
    rank: int
    score: float
    metadata: Dict[str, Any] = {}


class ModelInfo(BaseModel):
    """Model information in response."""
    retrieval_model: str
    generation_model: str
    reranker_model: str


class QueryResponse(BaseModel):
    """Response schema for POST /query."""
    response_id: str
    answer: str
    sources: List[SourceResponse]
    confidence: float
    model_info: ModelInfo
    latency_ms: float
    session_id: Optional[str] = None


class QueryHistoryResponse(BaseModel):
    """Response schema for GET /query/:response_id."""
    response_id: str
    query: str
    answer: str
    sources: List[SourceResponse]
    confidence: float
    created_at: datetime
    feedback_summary: Optional[Dict[str, Any]] = None


# ==================== Feedback Schemas ====================

class FeedbackTypeEnum(str, Enum):
    """Feedback types."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CLICK = "click"
    DWELL = "dwell"
    COPY = "copy"
    SHARE = "share"


class FeedbackReasonCode(str, Enum):
    """Reason codes for negative feedback."""
    WRONG_ANSWER = "wrong_answer"
    OUTDATED = "outdated"
    IRRELEVANT = "irrelevant"
    INCOMPLETE = "incomplete"
    OTHER = "other"


class FeedbackReason(BaseModel):
    """Reason for feedback."""
    code: Optional[FeedbackReasonCode] = None
    text: Optional[str] = Field(None, max_length=1000)


class FeedbackRequest(BaseModel):
    """Request schema for POST /feedback."""
    response_id: str
    type: FeedbackTypeEnum
    value: Optional[float] = Field(None, ge=1, le=5)
    target_chunk_id: Optional[str] = None
    reason: Optional[FeedbackReason] = None
    client_timestamp: Optional[datetime] = None


class FeedbackResponse(BaseModel):
    """Response schema for POST /feedback."""
    feedback_id: str
    accepted: bool
    message: str = ""


class BatchFeedbackEvent(BaseModel):
    """A single event in batch feedback."""
    response_id: str
    type: FeedbackTypeEnum
    value: Optional[float] = None
    target_chunk_id: Optional[str] = None
    client_timestamp: Optional[datetime] = None


class BatchFeedbackRequest(BaseModel):
    """Request schema for POST /feedback/batch."""
    events: List[BatchFeedbackEvent] = Field(..., max_length=100)


class BatchFeedbackResponse(BaseModel):
    """Response schema for POST /feedback/batch."""
    accepted: int
    rejected: int
    errors: List[Dict[str, str]] = []


# ==================== Document Schemas ====================

class ChunkingConfig(BaseModel):
    """Chunking configuration for documents."""
    strategy: str = Field(default="semantic", pattern="^(semantic|fixed|sentence)$")
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class DocumentRequest(BaseModel):
    """Request schema for POST /documents."""
    external_id: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source_type: str = Field(..., pattern="^(upload|crawl|api|manual)$")
    source_url: Optional[str] = None
    language: str = Field(default="en", min_length=2, max_length=5)
    metadata: Optional[Dict[str, Any]] = None
    chunking: Optional[ChunkingConfig] = None

    @field_validator('content')
    @classmethod
    def validate_content_size(cls, v):
        if len(v.encode('utf-8')) > 10 * 1024 * 1024:
            raise ValueError('Content exceeds 10MB limit')
        return v


class DocumentResponse(BaseModel):
    """Response schema for POST /documents."""
    document_id: str
    external_id: Optional[str] = None
    chunks_created: int
    status: str
    estimated_completion: Optional[datetime] = None


class DocumentAsyncResponse(BaseModel):
    """Response for async document processing."""
    document_id: str
    status: str
    job_id: str
    poll_url: str


class DocumentDetailResponse(BaseModel):
    """Response schema for GET /documents/:document_id."""
    document_id: str
    external_id: Optional[str] = None
    title: str
    source_type: str
    source_url: Optional[str] = None
    language: str
    metadata: Dict[str, Any] = {}
    status: str
    chunks_count: int
    created_at: datetime
    updated_at: datetime
    embedding_model: Optional[str] = None


class DocumentListItem(BaseModel):
    """A document in the list response."""
    document_id: str
    external_id: Optional[str] = None
    title: str
    source_type: str
    status: str
    chunks_count: int
    created_at: datetime
    updated_at: datetime


class Pagination(BaseModel):
    """Pagination information."""
    page: int
    per_page: int
    total: int
    total_pages: int


class DocumentListResponse(BaseModel):
    """Response schema for GET /documents."""
    documents: List[DocumentListItem]
    pagination: Pagination


# ==================== Admin Schemas ====================

class RetrievalMetrics(BaseModel):
    """Retrieval performance metrics."""
    mrr: float
    ndcg_10: float
    recall_10: float
    avg_latency_ms: float
    p95_latency_ms: float


class FeedbackMetrics(BaseModel):
    """Feedback statistics."""
    total_events: int
    thumbs_up_rate: float
    thumbs_down_rate: float
    avg_rating: float
    click_through_rate: float


class UsageMetrics(BaseModel):
    """Usage statistics."""
    total_queries: int
    unique_sessions: int
    queries_per_session: float


class SystemMetrics(BaseModel):
    """System performance metrics."""
    vector_store_latency_p95_ms: float
    embedding_latency_p95_ms: float
    generation_latency_p95_ms: float
    error_rate: float


class LearningMetrics(BaseModel):
    """Learning pipeline metrics."""
    last_training: Optional[datetime] = None
    training_samples: int
    model_version: str
    improvement_vs_baseline: float


class MetricsResponse(BaseModel):
    """Response schema for GET /admin/metrics."""
    period: str
    retrieval: RetrievalMetrics
    feedback: FeedbackMetrics
    usage: UsageMetrics
    system: SystemMetrics
    learning: LearningMetrics


class ModelVersionResponse(BaseModel):
    """A model version."""
    version: str
    type: str
    status: str
    traffic_percent: float
    created_at: datetime
    metrics: Dict[str, float]


class ModelListResponse(BaseModel):
    """Response schema for GET /admin/models."""
    models: List[ModelVersionResponse]


class DeployRequest(BaseModel):
    """Request schema for POST /admin/models/:version/deploy."""
    strategy: str = Field(..., pattern="^(immediate|canary|blue_green)$")
    traffic_percent: float = Field(default=10, ge=1, le=100)
    auto_promote: bool = True
    rollback_threshold: float = Field(default=0.95, ge=0, le=1)


class DeployResponse(BaseModel):
    """Response for model deployment."""
    deployment_id: str
    status: str
    model_version: str
    strategy: str
    traffic_percent: float
    started_at: datetime


class RollbackRequest(BaseModel):
    """Request schema for POST /admin/models/:version/rollback."""
    reason: str = Field(..., min_length=1, max_length=500)
    target_version: Optional[str] = None


class RollbackResponse(BaseModel):
    """Response for model rollback."""
    rollback_id: str
    from_version: str
    to_version: str
    status: str
    started_at: datetime


class LearningStatusResponse(BaseModel):
    """Response schema for GET /admin/learning/status."""
    status: str
    current_job: Optional[Dict[str, Any]] = None
    last_completed: Optional[Dict[str, Any]] = None
    queue: Dict[str, int]
    schedule: Dict[str, str]


class TriggerTrainingRequest(BaseModel):
    """Request schema for POST /admin/learning/trigger."""
    force: bool = False
    config_override: Optional[Dict[str, Any]] = None


class TriggerTrainingResponse(BaseModel):
    """Response for triggering training."""
    job_id: str
    status: str
    estimated_start: Optional[datetime] = None


class ExperimentResponse(BaseModel):
    """A/B experiment response."""
    id: str
    name: str
    status: str
    control: str
    treatment: str
    traffic_percent: float
    started_at: datetime
    metrics: Dict[str, Any]


class ExperimentListResponse(BaseModel):
    """Response schema for GET /admin/experiments."""
    experiments: List[ExperimentResponse]


class CreateExperimentRequest(BaseModel):
    """Request schema for POST /admin/experiments."""
    name: str = Field(..., min_length=1, max_length=100)
    control_model: str
    treatment_model: str
    traffic_percent: float = Field(default=10, ge=1, le=50)
    primary_metric: str = "mrr"
    min_samples: int = Field(default=1000, ge=100)
    max_duration_hours: int = Field(default=168, ge=1)


class CreateExperimentResponse(BaseModel):
    """Response for experiment creation."""
    experiment_id: str
    status: str
    started_at: datetime


class StopExperimentRequest(BaseModel):
    """Request schema for POST /admin/experiments/:id/stop."""
    winner: Optional[str] = Field(None, pattern="^(control|treatment|none)$")
    reason: str = Field(..., min_length=1, max_length=500)


class StopExperimentResponse(BaseModel):
    """Response for stopping experiment."""
    experiment_id: str
    status: str
    winner: Optional[str] = None
    stopped_at: datetime


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: Dict[str, bool]
