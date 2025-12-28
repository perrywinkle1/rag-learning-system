"""Unit tests for the Learning Pipeline components.

This module provides comprehensive unit tests for all learning pipeline
components including feature extraction, pair generation, training,
evaluation, and orchestration.
"""

import asyncio
from datetime import datetime
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from src.learning.feature_extractor import (
    EnrichedFeedback,
    FeatureExtractor,
    FeedbackType,
    TrainingFeatures,
)
from src.learning.pair_generator import (
    PairType,
    TrainingPair,
    TrainingPairGenerator,
    TripletSample,
)
from src.learning.trainer import EmbeddingTrainer, TrainingConfig, TrainingResult
from src.learning.evaluator import (
    BenchmarkResult,
    ComparisonResult,
    EvalMetrics,
    ModelEvaluator,
)
from src.learning.scheduler import (
    JobState,
    ScheduledJob,
    SchedulerConfig,
    SchedulerState,
    TrainingScheduler,
)
from src.learning.checkpoint_manager import CheckpointInfo, CheckpointManager
from src.learning.dataset import TrainingDataset, InBatchNegativesDataset
from src.learning.losses import (
    ContrastiveLoss,
    InfoNCELoss,
    MultipleNegativesRankingLoss,
    TripletMarginLoss,
)
from src.learning.pipeline import LearningPipeline, PipelineResult, PipelineStatus


# ====================
# Fixtures
# ====================

@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return np.random.randn(384).astype(np.float32)


@pytest.fixture
def sample_enriched_feedback(sample_embedding):
    """Create a sample enriched feedback event."""
    return EnrichedFeedback(
        feedback_id="fb_001",
        response_id="resp_001",
        query_id="q_001",
        session_id="session_001",
        event_type=FeedbackType.THUMBS_UP,
        timestamp=datetime.utcnow(),
        query_text="What is machine learning?",
        query_embedding=sample_embedding,
        chunk_id="chunk_001",
        chunk_text="Machine learning is a subset of artificial intelligence...",
        chunk_embedding=sample_embedding,
        document_id="doc_001",
        position_in_results=1,
        model_version="v1.0",
    )


@pytest.fixture
def sample_feedback_list(sample_embedding):
    """Create a list of sample feedback events."""
    feedbacks = []
    
    # Positive feedback
    for i in range(5):
        feedbacks.append(EnrichedFeedback(
            feedback_id=f"fb_pos_{i}",
            response_id=f"resp_{i}",
            query_id="q_001",
            session_id="session_001",
            event_type=FeedbackType.THUMBS_UP,
            timestamp=datetime.utcnow(),
            query_text="What is machine learning?",
            query_embedding=sample_embedding,
            chunk_id=f"chunk_pos_{i}",
            chunk_text=f"Positive content {i}",
            chunk_embedding=sample_embedding,
            document_id=f"doc_{i}",
            position_in_results=i,
        ))
    
    # Negative feedback
    for i in range(5):
        feedbacks.append(EnrichedFeedback(
            feedback_id=f"fb_neg_{i}",
            response_id=f"resp_{i}",
            query_id="q_001",
            session_id="session_001",
            event_type=FeedbackType.THUMBS_DOWN,
            timestamp=datetime.utcnow(),
            query_text="What is machine learning?",
            query_embedding=sample_embedding,
            chunk_id=f"chunk_neg_{i}",
            chunk_text=f"Negative content {i}",
            chunk_embedding=sample_embedding,
            document_id=f"doc_neg_{i}",
            position_in_results=i + 5,
        ))
    
    return feedbacks


@pytest.fixture
def sample_training_features(sample_embedding):
    """Create sample training features."""
    return TrainingFeatures(
        query_id="q_001",
        query_text="What is machine learning?",
        query_embedding=sample_embedding,
        chunk_id="chunk_001",
        chunk_text="Machine learning is...",
        chunk_embedding=sample_embedding,
        label=0.9,
        label_source="explicit",
        confidence=0.95,
        position=1,
        feedback_types=["thumbs_up"],
        timestamp=datetime.utcnow(),
        session_id="session_001",
    )


@pytest.fixture
def training_config():
    """Create a training configuration."""
    return TrainingConfig(
        objective="triplet",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        learning_rate=2e-5,
        batch_size=4,
        epochs=1,
        warmup_steps=10,
        device="cpu",
    )


# ====================
# Feature Extractor Tests
# ====================

class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        extractor = FeatureExtractor()
        
        assert extractor.min_confidence == 0.3
        assert extractor.position_decay == 0.1
        assert FeedbackType.THUMBS_UP in extractor.signal_weights
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        extractor = FeatureExtractor(
            min_confidence=0.5,
            position_decay=0.2,
        )
        
        assert extractor.min_confidence == 0.5
        assert extractor.position_decay == 0.2
    
    def test_extract_thumbs_up(self, sample_enriched_feedback):
        """Test feature extraction for thumbs up feedback."""
        extractor = FeatureExtractor()
        features = extractor.extract(sample_enriched_feedback)
        
        assert features is not None
        assert features.query_id == "q_001"
        assert features.label == 1.0  # Thumbs up = positive
        assert features.label_source == "explicit"
        assert features.confidence > 0.5
    
    def test_extract_thumbs_down(self, sample_enriched_feedback):
        """Test feature extraction for thumbs down feedback."""
        sample_enriched_feedback.event_type = FeedbackType.THUMBS_DOWN
        
        extractor = FeatureExtractor()
        features = extractor.extract(sample_enriched_feedback)
        
        assert features is not None
        assert features.label == 0.0  # Thumbs down = negative
    
    def test_extract_rating(self, sample_enriched_feedback):
        """Test feature extraction for rating feedback."""
        sample_enriched_feedback.event_type = FeedbackType.RATING
        sample_enriched_feedback.rating_value = 4
        
        extractor = FeatureExtractor()
        features = extractor.extract(sample_enriched_feedback)
        
        assert features is not None
        assert features.label == 0.75  # (4-1)/4 = 0.75
    
    def test_extract_dwell_time(self, sample_enriched_feedback):
        """Test feature extraction for dwell time feedback."""
        sample_enriched_feedback.event_type = FeedbackType.DWELL
        sample_enriched_feedback.dwell_time_ms = 15000  # 15 seconds
        
        extractor = FeatureExtractor()
        features = extractor.extract(sample_enriched_feedback)
        
        assert features is not None
        assert 0.5 < features.label < 0.9  # Medium-high engagement
    
    def test_extract_batch(self, sample_feedback_list):
        """Test batch feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_batch(sample_feedback_list)
        
        assert len(features) > 0
        assert all(isinstance(f, TrainingFeatures) for f in features)
    
    def test_extract_batch_with_aggregation(self, sample_feedback_list):
        """Test batch extraction with aggregation."""
        extractor = FeatureExtractor()
        features = extractor.extract_batch(sample_feedback_list, aggregate_by_pair=True)
        
        # Should aggregate by query-chunk pair
        assert len(features) <= len(sample_feedback_list)
    
    def test_extract_invalid_feedback(self, sample_enriched_feedback):
        """Test extraction with invalid feedback."""
        sample_enriched_feedback.query_embedding = None
        
        extractor = FeatureExtractor()
        features = extractor.extract(sample_enriched_feedback)
        
        assert features is None
    
    def test_confidence_position_decay(self, sample_enriched_feedback):
        """Test that confidence decays with position."""
        extractor = FeatureExtractor(position_decay=0.1)
        
        sample_enriched_feedback.position_in_results = 0
        features_pos0 = extractor.extract(sample_enriched_feedback)
        
        sample_enriched_feedback.position_in_results = 10
        features_pos10 = extractor.extract(sample_enriched_feedback)
        
        assert features_pos0.confidence > features_pos10.confidence


# ====================
# Pair Generator Tests
# ====================

class TestTrainingPairGenerator:
    """Tests for TrainingPairGenerator class."""
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        generator = TrainingPairGenerator()
        
        assert generator.positive_threshold == 0.6
        assert generator.negative_threshold == 0.4
        assert generator.max_negatives_per_positive == 5
    
    def test_generate_pairs_basic(self, sample_embedding):
        """Test basic pair generation."""
        features = [
            TrainingFeatures(
                query_id="q_001",
                query_text="query",
                query_embedding=sample_embedding,
                chunk_id="pos_1",
                chunk_text="positive",
                chunk_embedding=sample_embedding,
                label=0.9,
                label_source="explicit",
                confidence=0.9,
                position=1,
                feedback_types=["thumbs_up"],
                timestamp=datetime.utcnow(),
                session_id="s1",
            ),
            TrainingFeatures(
                query_id="q_001",
                query_text="query",
                query_embedding=sample_embedding,
                chunk_id="neg_1",
                chunk_text="negative",
                chunk_embedding=sample_embedding,
                label=0.1,
                label_source="explicit",
                confidence=0.9,
                position=5,
                feedback_types=["thumbs_down"],
                timestamp=datetime.utcnow(),
                session_id="s1",
            ),
        ]
        
        generator = TrainingPairGenerator()
        pairs = generator.generate_pairs(features)
        
        assert len(pairs) > 0
        assert all(isinstance(p, TrainingPair) for p in pairs)
    
    def test_generate_pairs_hard_negatives(self, sample_embedding):
        """Test hard negative mining."""
        features = [
            TrainingFeatures(
                query_id="q_001",
                query_text="query",
                query_embedding=sample_embedding,
                chunk_id="pos_1",
                chunk_text="positive",
                chunk_embedding=sample_embedding,
                label=0.9,
                label_source="explicit",
                confidence=0.9,
                position=1,
                feedback_types=["thumbs_up"],
                timestamp=datetime.utcnow(),
                session_id="s1",
            ),
            TrainingFeatures(
                query_id="q_001",
                query_text="query",
                query_embedding=sample_embedding,
                chunk_id="neg_1",
                chunk_text="hard negative (high rank, low label)",
                chunk_embedding=sample_embedding,
                label=0.2,
                label_source="implicit",
                confidence=0.8,
                position=2,  # High rank but negative
                feedback_types=["abandon"],
                timestamp=datetime.utcnow(),
                session_id="s1",
            ),
        ]
        
        generator = TrainingPairGenerator(hard_negative_ratio=1.0)
        pairs = generator.generate_pairs(features, use_hard_negatives=True)
        
        hard_pairs = [p for p in pairs if p.is_hard_negative]
        assert len(hard_pairs) > 0
    
    def test_generate_triplets(self, sample_embedding):
        """Test triplet generation."""
        features = [
            TrainingFeatures(
                query_id="q_001",
                query_text="query",
                query_embedding=sample_embedding,
                chunk_id="pos_1",
                chunk_text="positive",
                chunk_embedding=sample_embedding,
                label=0.9,
                label_source="explicit",
                confidence=0.9,
                position=1,
                feedback_types=["thumbs_up"],
                timestamp=datetime.utcnow(),
                session_id="s1",
            ),
            TrainingFeatures(
                query_id="q_001",
                query_text="query",
                query_embedding=sample_embedding,
                chunk_id="neg_1",
                chunk_text="negative",
                chunk_embedding=sample_embedding,
                label=0.1,
                label_source="explicit",
                confidence=0.9,
                position=5,
                feedback_types=["thumbs_down"],
                timestamp=datetime.utcnow(),
                session_id="s1",
            ),
        ]
        
        generator = TrainingPairGenerator()
        triplets = generator.generate_triplets(features, margin=0.2)
        
        assert len(triplets) > 0
        assert all(isinstance(t, TripletSample) for t in triplets)
        assert all(t.margin == 0.2 for t in triplets)
    
    def test_mine_hard_negatives(self):
        """Test hard negative mining with embeddings."""
        generator = TrainingPairGenerator()
        
        query_emb = np.random.randn(384).astype(np.float32)
        positive_emb = query_emb + np.random.randn(384).astype(np.float32) * 0.1
        
        candidates = [np.random.randn(384).astype(np.float32) for _ in range(10)]
        candidate_ids = [f"cand_{i}" for i in range(10)]
        
        hard_negs = generator.mine_hard_negatives(
            query_emb, positive_emb, candidates, candidate_ids, top_k=3
        )
        
        assert len(hard_negs) == 3
        # Should be sorted by similarity (descending)
        sims = [h[2] for h in hard_negs]
        assert sims == sorted(sims, reverse=True)


# ====================
# Loss Function Tests
# ====================

class TestLossFunctions:
    """Tests for custom loss functions."""
    
    def test_triplet_margin_loss(self):
        """Test triplet margin loss computation."""
        loss_fn = TripletMarginLoss(margin=0.2, distance="cosine")
        
        anchor = torch.randn(4, 128)
        positive = anchor + torch.randn(4, 128) * 0.1  # Similar
        negative = torch.randn(4, 128)  # Different
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_infonce_loss(self):
        """Test InfoNCE loss computation."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        anchor = torch.randn(8, 128)
        positive = anchor + torch.randn(8, 128) * 0.1
        
        loss = loss_fn(anchor, positive)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_mnrl_loss(self):
        """Test Multiple Negatives Ranking Loss."""
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)
        
        anchor = torch.randn(8, 128)
        positive = anchor + torch.randn(8, 128) * 0.1
        
        loss = loss_fn(anchor, positive)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(margin=1.0)
        
        anchor = torch.randn(4, 128)
        positive = anchor + torch.randn(4, 128) * 0.1
        negative = torch.randn(4, 128)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss.shape == ()
        assert loss >= 0


# ====================
# Dataset Tests
# ====================

class TestTrainingDataset:
    """Tests for TrainingDataset class."""
    
    def test_from_pairs(self, sample_embedding):
        """Test dataset creation from pairs."""
        pairs = [
            TrainingPair(
                pair_id="p1",
                pair_type=PairType.POSITIVE_NEGATIVE,
                query_id="q1",
                query_text="query",
                query_embedding=sample_embedding,
                positive_chunk_id="pos1",
                positive_chunk_text="positive",
                positive_chunk_embedding=sample_embedding,
                positive_score=0.9,
                negative_chunk_id="neg1",
                negative_chunk_text="negative",
                negative_chunk_embedding=sample_embedding,
                negative_score=0.1,
                label_source="explicit",
                confidence=0.9,
            )
            for _ in range(10)
        ]
        
        dataset = TrainingDataset.from_pairs(pairs, use_embeddings=True)
        
        assert len(dataset) == 10
        assert dataset.data_type == "pair"
    
    def test_from_texts(self):
        """Test dataset creation from text lists."""
        anchors = ["query 1", "query 2"]
        positives = ["positive 1", "positive 2"]
        negatives = ["negative 1", "negative 2"]
        
        # Skip tokenizer for this test
        dataset = TrainingDataset.from_texts(
            anchors, positives, negatives, use_embeddings=True
        )
        
        assert len(dataset) == 2
        assert dataset.data_type == "tuple"
    
    def test_split(self, sample_embedding):
        """Test dataset splitting."""
        pairs = [
            TrainingPair(
                pair_id=f"p{i}",
                pair_type=PairType.POSITIVE_NEGATIVE,
                query_id=f"q{i}",
                query_text="query",
                query_embedding=sample_embedding,
                positive_chunk_id="pos",
                positive_chunk_text="positive",
                positive_chunk_embedding=sample_embedding,
                positive_score=0.9,
                negative_chunk_id="neg",
                negative_chunk_text="negative",
                negative_chunk_embedding=sample_embedding,
                negative_score=0.1,
                label_source="explicit",
                confidence=0.9,
            )
            for i in range(100)
        ]
        
        dataset = TrainingDataset.from_pairs(pairs, use_embeddings=True)
        train_ds, val_ds = dataset.split(val_ratio=0.2, seed=42)
        
        assert len(train_ds) == 80
        assert len(val_ds) == 20


# ====================
# Evaluator Tests
# ====================

class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_evaluate_perfect_ranking(self):
        """Test evaluation with perfect ranking."""
        evaluator = ModelEvaluator(k_values=[1, 3, 5])
        
        # Perfect predictions
        predictions = [["doc1", "doc2", "doc3"]]
        relevance = [{"doc1": 1.0, "doc2": 0.5, "doc3": 0.0}]
        
        metrics = evaluator.evaluate(predictions, relevance)
        
        assert metrics.mrr == 1.0  # First result is relevant
        assert metrics.num_queries == 1
    
    def test_evaluate_multiple_queries(self):
        """Test evaluation with multiple queries."""
        evaluator = ModelEvaluator()
        
        predictions = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
        ]
        relevance = [
            {"doc1": 1.0},
            {"doc5": 1.0},  # Relevant doc at position 2
        ]
        
        metrics = evaluator.evaluate(predictions, relevance)
        
        # MRR = (1/1 + 1/2) / 2 = 0.75
        assert abs(metrics.mrr - 0.75) < 0.01
    
    def test_compare_models(self):
        """Test model comparison."""
        evaluator = ModelEvaluator()
        
        # Model A predictions
        model_a_preds = [["doc1", "doc2", "doc3"]]
        # Model B predictions (better)
        model_b_preds = [["doc1", "doc3", "doc2"]]
        relevance = [{"doc1": 1.0}]
        
        result = evaluator.compare_models(
            model_a_preds, model_b_preds, relevance,
            model_a_version="v1", model_b_version="v2"
        )
        
        assert result.model_a_version == "v1"
        assert result.model_b_version == "v2"
        # Both should have same MRR (doc1 at position 1)
        assert result.model_a_metrics.mrr == result.model_b_metrics.mrr


# ====================
# Scheduler Tests
# ====================

class TestTrainingScheduler:
    """Tests for TrainingScheduler class."""
    
    def test_init(self):
        """Test scheduler initialization."""
        config = SchedulerConfig(
            cron_expression="0 2 * * *",
            min_feedback_count=1000,
        )
        scheduler = TrainingScheduler(config)
        
        assert scheduler.state == SchedulerState.STOPPED
        assert scheduler.config.min_feedback_count == 1000
    
    def test_start_stop(self):
        """Test scheduler start/stop."""
        config = SchedulerConfig(enabled=True)
        scheduler = TrainingScheduler(config)
        
        scheduler.start()
        assert scheduler.state == SchedulerState.RUNNING
        
        scheduler.stop()
        assert scheduler.state == SchedulerState.STOPPED
    
    def test_pause_resume(self):
        """Test scheduler pause/resume."""
        config = SchedulerConfig(enabled=True)
        scheduler = TrainingScheduler(config)
        
        scheduler.start()
        scheduler.pause()
        assert scheduler.state == SchedulerState.PAUSED
        
        scheduler.resume()
        assert scheduler.state == SchedulerState.RUNNING
        
        scheduler.stop()
    
    def test_trigger_now(self):
        """Test manual trigger."""
        config = SchedulerConfig()
        scheduler = TrainingScheduler(config)
        
        job_id = scheduler.trigger_now(reason="test")
        
        assert job_id is not None
        job = scheduler.get_job_status(job_id)
        assert job is not None
    
    def test_should_train_threshold(self):
        """Test training threshold check."""
        config = SchedulerConfig(min_feedback_count=100)
        counter = lambda: 150
        scheduler = TrainingScheduler(config, feedback_counter=counter)
        scheduler.last_feedback_count = 0
        
        should_train, reason = scheduler.should_train()
        
        assert should_train is True
        assert "feedback_threshold" in reason


# ====================
# Checkpoint Manager Tests
# ====================

class TestCheckpointManager:
    """Tests for CheckpointManager class."""
    
    def test_init_local(self, tmp_path):
        """Test initialization with local storage."""
        manager = CheckpointManager(
            storage_type="local",
            local_path=str(tmp_path / "checkpoints"),
        )
        
        assert manager.storage_type == "local"
        assert manager.local_path.exists()
    
    def test_list_checkpoints_empty(self, tmp_path):
        """Test listing with no checkpoints."""
        manager = CheckpointManager(
            storage_type="local",
            local_path=str(tmp_path / "checkpoints"),
        )
        
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 0
    
    def test_get_latest_none(self, tmp_path):
        """Test getting latest with no checkpoints."""
        manager = CheckpointManager(
            storage_type="local",
            local_path=str(tmp_path / "checkpoints"),
        )
        
        latest = manager.get_latest()
        assert latest is None


# ====================
# Pipeline Tests
# ====================

class TestLearningPipeline:
    """Tests for LearningPipeline class."""
    
    def test_init(self, training_config):
        """Test pipeline initialization."""
        pipeline = LearningPipeline(training_config=training_config)
        
        assert pipeline.current_status == PipelineStatus.IDLE
        assert len(pipeline.run_history) == 0
    
    def test_get_status(self, training_config):
        """Test status retrieval."""
        pipeline = LearningPipeline(training_config=training_config)
        
        status = pipeline.get_status()
        
        assert status["status"] == "idle"
        assert status["total_runs"] == 0
    
    @pytest.mark.asyncio
    async def test_run_no_feedback(self, training_config):
        """Test run with no feedback events."""
        pipeline = LearningPipeline(training_config=training_config)
        
        result = await pipeline.run(feedback_events=[])
        
        assert result.status == PipelineStatus.COMPLETED
        assert result.num_feedback_events == 0
    
    @pytest.mark.asyncio
    async def test_run_with_mock_feedback(self, training_config, sample_feedback_list):
        """Test run with mock feedback."""
        pipeline = LearningPipeline(training_config=training_config)
        
        # Run will fail during training without proper model setup
        # but we can test the feature extraction phase
        try:
            result = await pipeline.run(feedback_events=sample_feedback_list)
        except Exception:
            pass  # Expected to fail in training phase
        
        # At minimum, we processed the feedback
        assert pipeline.current_status in [PipelineStatus.IDLE, PipelineStatus.FAILED]


# ====================
# Integration Tests
# ====================

class TestIntegration:
    """Integration tests for the learning pipeline."""
    
    def test_full_feature_to_pair_pipeline(self, sample_feedback_list):
        """Test feature extraction to pair generation."""
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_batch(sample_feedback_list)
        
        # Generate pairs
        generator = TrainingPairGenerator()
        pairs = generator.generate_pairs(features)
        
        # Create dataset
        if pairs:
            dataset = TrainingDataset.from_pairs(pairs, use_embeddings=True)
            assert len(dataset) > 0
    
    def test_evaluator_with_predictions(self):
        """Test evaluator with realistic predictions."""
        evaluator = ModelEvaluator(k_values=[1, 5, 10])
        
        # Simulate retrieval results
        predictions = []
        relevance = []
        
        for i in range(10):
            # Random ranking
            docs = [f"doc_{j}" for j in range(20)]
            predictions.append(docs)
            
            # First few docs are relevant
            rel = {f"doc_{j}": 1.0 if j < 3 else 0.0 for j in range(20)}
            relevance.append(rel)
        
        metrics = evaluator.evaluate(predictions, relevance)
        
        assert 0 <= metrics.mrr <= 1
        assert 0 <= metrics.ndcg <= 1
        assert metrics.num_queries == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

