"""Admin endpoints for system management."""

import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import (
    MetricsResponse,
    RetrievalMetrics,
    FeedbackMetrics,
    UsageMetrics,
    SystemMetrics,
    LearningMetrics,
    ModelListResponse,
    ModelVersionResponse,
    DeployRequest,
    DeployResponse,
    RollbackRequest,
    RollbackResponse,
    LearningStatusResponse,
    TriggerTrainingRequest,
    TriggerTrainingResponse,
    ExperimentListResponse,
    ExperimentResponse,
    CreateExperimentRequest,
    CreateExperimentResponse,
    StopExperimentRequest,
    StopExperimentResponse,
)
from src.api.dependencies import get_settings_dep
from src.config import Settings

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    period: str = Query(default="24h", pattern="^(1h|24h|7d|30d)$"),
    settings: Settings = Depends(get_settings_dep),
):
    """Get system metrics and health indicators.

    Returns retrieval performance, feedback statistics, usage data,
    system health, and learning pipeline status.
    """
    # In production, these would come from Prometheus/metrics store
    return MetricsResponse(
        period=period,
        retrieval=RetrievalMetrics(
            mrr=0.72,
            ndcg_10=0.68,
            recall_10=0.85,
            avg_latency_ms=156,
            p95_latency_ms=312,
        ),
        feedback=FeedbackMetrics(
            total_events=15234,
            thumbs_up_rate=0.78,
            thumbs_down_rate=0.12,
            avg_rating=4.1,
            click_through_rate=0.45,
        ),
        usage=UsageMetrics(
            total_queries=45678,
            unique_sessions=12345,
            queries_per_session=3.7,
        ),
        system=SystemMetrics(
            vector_store_latency_p95_ms=45,
            embedding_latency_p95_ms=12,
            generation_latency_p95_ms=2100,
            error_rate=0.002,
        ),
        learning=LearningMetrics(
            last_training=datetime.utcnow(),
            training_samples=50000,
            model_version="v1.0.0",
            improvement_vs_baseline=0.08,
        ),
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all model versions with their status."""
    # In production, fetch from model registry
    return ModelListResponse(
        models=[
            ModelVersionResponse(
                version="v1.0.0",
                type="embedding",
                status="active",
                traffic_percent=100,
                created_at=datetime.utcnow(),
                metrics={"mrr": 0.72, "ndcg_10": 0.68},
            ),
        ]
    )


@router.post("/models/{version}/deploy", response_model=DeployResponse)
async def deploy_model(
    version: str,
    request: DeployRequest,
):
    """Deploy a model version to production.

    Supports different deployment strategies:
    - immediate: Replace current model immediately
    - canary: Gradual rollout to percentage of traffic
    - blue_green: Full switch with instant rollback capability
    """
    return DeployResponse(
        deployment_id=str(uuid.uuid4()),
        status="deploying",
        model_version=version,
        strategy=request.strategy,
        traffic_percent=request.traffic_percent,
        started_at=datetime.utcnow(),
    )


@router.post("/models/{version}/rollback", response_model=RollbackResponse)
async def rollback_model(
    version: str,
    request: RollbackRequest,
):
    """Rollback to a previous model version."""
    return RollbackResponse(
        rollback_id=str(uuid.uuid4()),
        from_version=version,
        to_version=request.target_version or "v0.9.0",
        status="rolling_back",
        started_at=datetime.utcnow(),
    )


@router.get("/learning/status", response_model=LearningStatusResponse)
async def get_learning_status():
    """Get current learning pipeline status."""
    return LearningStatusResponse(
        status="active",
        current_job=None,
        last_completed={
            "job_id": str(uuid.uuid4()),
            "completed_at": datetime.utcnow().isoformat(),
            "result": "success",
            "model_version": "v1.0.0",
        },
        queue={
            "pending_feedback_events": 15234,
            "training_pairs_available": 45678,
        },
        schedule={
            "next_training": datetime.utcnow().isoformat(),
            "frequency": "daily",
        },
    )


@router.post("/learning/pause")
async def pause_learning():
    """Pause the learning pipeline."""
    return {
        "status": "paused",
        "paused_at": datetime.utcnow().isoformat(),
        "message": "Learning pipeline paused. Feedback collection continues.",
    }


@router.post("/learning/resume")
async def resume_learning():
    """Resume the learning pipeline."""
    return {
        "status": "active",
        "resumed_at": datetime.utcnow().isoformat(),
        "message": "Learning pipeline resumed.",
    }


@router.post("/learning/trigger", response_model=TriggerTrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_training(
    request: TriggerTrainingRequest,
):
    """Manually trigger a training job."""
    return TriggerTrainingResponse(
        job_id=str(uuid.uuid4()),
        status="queued",
        estimated_start=datetime.utcnow(),
    )


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments():
    """List A/B experiments."""
    return ExperimentListResponse(
        experiments=[
            ExperimentResponse(
                id=str(uuid.uuid4()),
                name="embedding-v1.1-test",
                status="running",
                control="v1.0.0",
                treatment="v1.1.0",
                traffic_percent=10,
                started_at=datetime.utcnow(),
                metrics={
                    "control_mrr": 0.72,
                    "treatment_mrr": 0.74,
                    "p_value": 0.08,
                    "samples": {"control": 5432, "treatment": 603},
                },
            ),
        ]
    )


@router.post("/experiments", response_model=CreateExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    request: CreateExperimentRequest,
):
    """Create a new A/B experiment."""
    return CreateExperimentResponse(
        experiment_id=str(uuid.uuid4()),
        status="running",
        started_at=datetime.utcnow(),
    )


@router.post("/experiments/{experiment_id}/stop", response_model=StopExperimentResponse)
async def stop_experiment(
    experiment_id: str,
    request: StopExperimentRequest,
):
    """Stop an experiment early."""
    return StopExperimentResponse(
        experiment_id=experiment_id,
        status="stopped",
        winner=request.winner,
        stopped_at=datetime.utcnow(),
    )
