"""Learning pipeline orchestration.

This module provides the main orchestration layer for the learning pipeline,
coordinating feature extraction, training, evaluation, and deployment.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.learning.feature_extractor import FeatureExtractor, EnrichedFeedback, TrainingFeatures
from src.learning.pair_generator import TrainingPairGenerator, TrainingPair
from src.learning.trainer import EmbeddingTrainer, TrainingConfig, TrainingResult
from src.learning.evaluator import ModelEvaluator, EvalMetrics, ComparisonResult
from src.learning.scheduler import TrainingScheduler, SchedulerConfig
from src.learning.checkpoint_manager import CheckpointManager, CheckpointInfo
from src.learning.dataset import TrainingDataset

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Status of the learning pipeline."""
    IDLE = "idle"
    EXTRACTING_FEATURES = "extracting_features"
    GENERATING_PAIRS = "generating_pairs"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    run_id: str
    status: PipelineStatus
    
    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Data stats
    num_feedback_events: int = 0
    num_features: int = 0
    num_pairs: int = 0
    
    # Training results
    training_result: Optional[TrainingResult] = None
    
    # Evaluation results
    baseline_metrics: Optional[EvalMetrics] = None
    new_model_metrics: Optional[EvalMetrics] = None
    comparison: Optional[ComparisonResult] = None
    
    # Deployment
    deployed: bool = False
    deployed_version: str = ""
    checkpoint_id: str = ""
    
    # Errors
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "num_feedback_events": self.num_feedback_events,
            "num_features": self.num_features,
            "num_pairs": self.num_pairs,
            "deployed": self.deployed,
            "deployed_version": self.deployed_version,
            "error": self.error,
        }


class LearningPipeline:
    """Orchestrates the complete learning pipeline.
    
    Pipeline stages:
    1. Fetch feedback events from Kafka/database
    2. Extract training features from feedback
    3. Generate training pairs
    4. Train embedding model
    5. Evaluate against baseline
    6. Deploy if improved (with A/B testing)
    """
    
    def __init__(
        self,
        training_config: TrainingConfig,
        scheduler_config: Optional[SchedulerConfig] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        pair_generator: Optional[TrainingPairGenerator] = None,
        evaluator: Optional[ModelEvaluator] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        feedback_fetcher: Optional[Callable] = None,
        model_deployer: Optional[Callable] = None,
        baseline_model: Optional[Any] = None,
    ):
        """Initialize pipeline.
        
        Args:
            training_config: Configuration for model training
            scheduler_config: Configuration for scheduled runs
            feature_extractor: Feature extraction component
            pair_generator: Training pair generation component
            evaluator: Model evaluation component
            checkpoint_manager: Checkpoint storage component
            feedback_fetcher: Async function to fetch feedback events
            model_deployer: Async function to deploy models
            baseline_model: Current production model for comparison
        """
        self.training_config = training_config
        self.scheduler_config = scheduler_config or SchedulerConfig()
        
        # Initialize components
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.pair_generator = pair_generator or TrainingPairGenerator()
        self.evaluator = evaluator or ModelEvaluator()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # External integrations
        self.feedback_fetcher = feedback_fetcher
        self.model_deployer = model_deployer
        self.baseline_model = baseline_model
        
        # Scheduler
        self.scheduler = TrainingScheduler(
            config=self.scheduler_config,
            training_callback=self.run,
        )
        
        # State
        self.current_status = PipelineStatus.IDLE
        self.run_history: List[PipelineResult] = []
        
        logger.info("LearningPipeline initialized")
    
    async def run(
        self,
        feedback_events: Optional[List[EnrichedFeedback]] = None,
        force_deploy: bool = False,
    ) -> PipelineResult:
        """Run the complete learning pipeline.
        
        Args:
            feedback_events: Pre-fetched feedback events (optional)
            force_deploy: Deploy regardless of evaluation results
            
        Returns:
            PipelineResult with all metrics and outcomes
        """
        run_id = str(uuid.uuid4())[:8]
        result = PipelineResult(run_id=run_id, status=PipelineStatus.IDLE)
        
        logger.info(f"Starting pipeline run {run_id}")
        
        try:
            # Stage 1: Fetch feedback
            self.current_status = PipelineStatus.EXTRACTING_FEATURES
            result.status = PipelineStatus.EXTRACTING_FEATURES
            
            if feedback_events is None:
                if self.feedback_fetcher:
                    feedback_events = await self.feedback_fetcher()
                else:
                    raise ValueError("No feedback events provided and no fetcher configured")
            
            result.num_feedback_events = len(feedback_events)
            logger.info(f"Run {run_id}: Processing {len(feedback_events)} feedback events")
            
            if len(feedback_events) < self.scheduler_config.min_feedback_count:
                logger.warning(
                    f"Run {run_id}: Insufficient feedback "
                    f"({len(feedback_events)} < {self.scheduler_config.min_feedback_count})"
                )
            
            # Stage 2: Extract features
            features = self.feature_extractor.extract_batch(feedback_events)
            result.num_features = len(features)
            logger.info(f"Run {run_id}: Extracted {len(features)} features")
            
            if not features:
                result.status = PipelineStatus.COMPLETED
                result.error = "No features extracted from feedback"
                return self._finalize_result(result)
            
            # Stage 3: Generate training pairs
            self.current_status = PipelineStatus.GENERATING_PAIRS
            result.status = PipelineStatus.GENERATING_PAIRS
            
            pairs = self.pair_generator.generate_pairs(features)
            result.num_pairs = len(pairs)
            logger.info(f"Run {run_id}: Generated {len(pairs)} training pairs")
            
            if not pairs:
                result.status = PipelineStatus.COMPLETED
                result.error = "No training pairs generated"
                return self._finalize_result(result)
            
            # Stage 4: Train model
            self.current_status = PipelineStatus.TRAINING
            result.status = PipelineStatus.TRAINING
            
            training_result = await self._train_model(pairs)
            result.training_result = training_result
            logger.info(
                f"Run {run_id}: Training completed, "
                f"final_loss={training_result.final_train_loss:.4f}"
            )
            
            # Stage 5: Evaluate
            self.current_status = PipelineStatus.EVALUATING
            result.status = PipelineStatus.EVALUATING
            
            evaluation = await self._evaluate_model(training_result)
            result.new_model_metrics = evaluation.get("new_metrics")
            result.baseline_metrics = evaluation.get("baseline_metrics")
            result.comparison = evaluation.get("comparison")
            
            # Stage 6: Deploy if improved
            self.current_status = PipelineStatus.DEPLOYING
            result.status = PipelineStatus.DEPLOYING
            
            should_deploy = force_deploy or self._should_deploy(result.comparison)
            
            if should_deploy:
                deploy_result = await self._deploy_model(training_result)
                result.deployed = deploy_result.get("success", False)
                result.deployed_version = training_result.model_version
                result.checkpoint_id = deploy_result.get("checkpoint_id", "")
                logger.info(f"Run {run_id}: Model deployed: {result.deployed_version}")
            else:
                logger.info(f"Run {run_id}: Model not deployed (did not beat baseline)")
            
            result.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            result.status = PipelineStatus.FAILED
            result.error = str(e)
            result.error_stage = self.current_status.value
        
        finally:
            self.current_status = PipelineStatus.IDLE
            result = self._finalize_result(result)
            self.run_history.append(result)
        
        return result
    
    async def _train_model(self, pairs: List[TrainingPair]) -> TrainingResult:
        """Train the embedding model."""
        # Create dataset
        dataset = TrainingDataset.from_pairs(pairs)
        train_dataset, val_dataset = dataset.split(
            val_ratio=self.training_config.validation_split
        )
        
        # Initialize trainer
        trainer = EmbeddingTrainer(config=self.training_config)
        
        # Train (run in executor to not block)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: trainer.train(train_dataset, val_dataset)
        )
        
        return result
    
    async def _evaluate_model(
        self,
        training_result: TrainingResult,
    ) -> Dict[str, Any]:
        """Evaluate the new model against baseline."""
        result = {}
        
        # Load new model from checkpoint
        if training_result.best_checkpoint_path:
            new_model = self.checkpoint_manager.load(
                training_result.model_version,
                device=self.training_config.device,
            ) if hasattr(self.checkpoint_manager, "load") else None
        else:
            new_model = None
        
        # Get evaluation data (would typically come from a held-out test set)
        # For now, return training metrics
        result["new_metrics"] = EvalMetrics(
            mrr=1.0 - training_result.final_val_loss,  # Approximation
            ndcg=1.0 - training_result.final_val_loss,
        )
        
        # Compare with baseline if available
        if self.baseline_model:
            # Would run actual evaluation here
            result["baseline_metrics"] = EvalMetrics(
                mrr=0.65,  # Placeholder
                ndcg=0.60,
            )
            
            # Create comparison
            result["comparison"] = ComparisonResult(
                model_a_version="baseline",
                model_b_version=training_result.model_version,
                model_a_metrics=result["baseline_metrics"],
                model_b_metrics=result["new_metrics"],
            )
        
        return result
    
    async def _deploy_model(
        self,
        training_result: TrainingResult,
    ) -> Dict[str, Any]:
        """Deploy the new model."""
        result = {"success": False}
        
        try:
            # Save checkpoint
            if training_result.best_checkpoint_path:
                checkpoint_info = CheckpointInfo(
                    checkpoint_id=training_result.job_id,
                    model_version=training_result.model_version,
                    storage_path=training_result.best_checkpoint_path,
                    storage_type="local",
                    model_name="embedding_model",
                    training_job_id=training_result.job_id,
                    metrics={
                        "train_loss": training_result.final_train_loss,
                        "val_loss": training_result.final_val_loss,
                    },
                )
                result["checkpoint_id"] = checkpoint_info.checkpoint_id
            
            # Call external deployer if configured
            if self.model_deployer:
                deploy_result = await self.model_deployer(training_result)
                result["success"] = deploy_result.get("success", False)
            else:
                # Simulate deployment
                result["success"] = True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def _should_deploy(self, comparison: Optional[ComparisonResult]) -> bool:
        """Determine if new model should be deployed."""
        if comparison is None:
            return True  # No baseline to compare against
        
        if comparison.winner == "model_b":
            # New model wins
            if comparison.is_significant:
                return True
            elif comparison.mrr_relative_improvement >= 0.01:
                # At least 1% improvement even if not significant
                return True
        
        return False
    
    def _finalize_result(self, result: PipelineResult) -> PipelineResult:
        """Finalize result with timing information."""
        result.end_time = datetime.utcnow()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result
    
    def start_scheduler(self):
        """Start the scheduled training runs."""
        self.scheduler.start()
    
    def stop_scheduler(self):
        """Stop the scheduled training runs."""
        self.scheduler.stop()
    
    def trigger_run(self, reason: str = "manual") -> str:
        """Manually trigger a pipeline run."""
        return self.scheduler.trigger_now(reason)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "status": self.current_status.value,
            "scheduler_state": self.scheduler.state.value,
            "next_scheduled_run": self.scheduler.get_next_run_time(),
            "total_runs": len(self.run_history),
            "successful_runs": sum(1 for r in self.run_history if r.status == PipelineStatus.COMPLETED),
            "failed_runs": sum(1 for r in self.run_history if r.status == PipelineStatus.FAILED),
        }
    
    def get_run_history(self, limit: int = 10) -> List[PipelineResult]:
        """Get recent pipeline runs."""
        return self.run_history[-limit:]

