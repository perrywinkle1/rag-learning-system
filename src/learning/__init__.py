"""Learning Pipeline for Closed-Loop RAG System."""

from src.learning.feature_extractor import FeatureExtractor, TrainingFeatures
from src.learning.pair_generator import TrainingPairGenerator, TrainingPair
from src.learning.trainer import EmbeddingTrainer, TrainingConfig, TrainingResult
from src.learning.evaluator import ModelEvaluator, EvalMetrics, ComparisonResult, BenchmarkResult
from src.learning.scheduler import TrainingScheduler, SchedulerConfig
from src.learning.checkpoint_manager import CheckpointManager, CheckpointInfo
from src.learning.dataset import TrainingDataset
from src.learning.losses import TripletMarginLoss, InfoNCELoss, MultipleNegativesRankingLoss, ContrastiveLoss
from src.learning.pipeline import LearningPipeline, PipelineResult, PipelineStatus

__all__ = [
    "FeatureExtractor", "TrainingFeatures", "TrainingPairGenerator", "TrainingPair",
    "EmbeddingTrainer", "TrainingConfig", "TrainingResult", "ModelEvaluator",
    "EvalMetrics", "ComparisonResult", "BenchmarkResult", "TrainingScheduler",
    "SchedulerConfig", "CheckpointManager", "CheckpointInfo", "TrainingDataset",
    "TripletMarginLoss", "InfoNCELoss", "MultipleNegativesRankingLoss", "ContrastiveLoss",
    "LearningPipeline", "PipelineResult", "PipelineStatus",
]
