"""Feature extraction for training data generation.

This module extracts features from enriched feedback events to create
training signals for the embedding model.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback events."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CLICK = "click"
    DWELL = "dwell"
    COPY = "copy"
    SHARE = "share"
    ABANDON = "abandon"


@dataclass
class EnrichedFeedback:
    """Enriched feedback event with full context."""
    feedback_id: str
    response_id: str
    query_id: str
    session_id: str
    event_type: FeedbackType
    timestamp: datetime
    
    # Query context
    query_text: str
    query_embedding: Optional[np.ndarray] = None
    
    # Document context
    chunk_id: str = ""
    chunk_text: str = ""
    chunk_embedding: Optional[np.ndarray] = None
    document_id: str = ""
    
    # Feedback specifics
    rating_value: Optional[int] = None  # 1-5 for rating type
    dwell_time_ms: Optional[int] = None  # for dwell type
    position_in_results: Optional[int] = None
    
    # Additional context
    model_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingFeatures:
    """Features extracted for training."""
    query_id: str
    query_text: str
    query_embedding: np.ndarray
    
    chunk_id: str
    chunk_text: str
    chunk_embedding: np.ndarray
    
    # Signal strength and type
    label: float  # 0.0 to 1.0 relevance score
    label_source: str  # explicit, implicit, inferred
    confidence: float  # 0.0 to 1.0
    
    # Metadata for pair generation
    position: int
    feedback_types: List[str]
    timestamp: datetime
    session_id: str
    
    # Hash for deduplication
    feature_hash: str = ""
    
    def __post_init__(self):
        """Compute feature hash after initialization."""
        if not self.feature_hash:
            self.feature_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash for deduplication."""
        content = f"{self.query_id}:{self.chunk_id}:{self.label_source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class FeatureExtractor:
    """Extracts training features from enriched feedback events.
    
    This class processes feedback events and converts them into
    training features with appropriate labels and confidence scores.
    """
    
    # Signal weights for different feedback types
    DEFAULT_SIGNAL_WEIGHTS = {
        FeedbackType.THUMBS_UP: 1.0,
        FeedbackType.THUMBS_DOWN: 0.0,
        FeedbackType.RATING: None,  # Computed from value
        FeedbackType.CLICK: 0.7,
        FeedbackType.DWELL: None,  # Computed from duration
        FeedbackType.COPY: 0.9,
        FeedbackType.SHARE: 0.95,
        FeedbackType.ABANDON: 0.1,
    }
    
    # Confidence weights for different feedback types
    DEFAULT_CONFIDENCE_WEIGHTS = {
        FeedbackType.THUMBS_UP: 0.95,
        FeedbackType.THUMBS_DOWN: 0.95,
        FeedbackType.RATING: 0.9,
        FeedbackType.CLICK: 0.6,
        FeedbackType.DWELL: 0.5,
        FeedbackType.COPY: 0.85,
        FeedbackType.SHARE: 0.9,
        FeedbackType.ABANDON: 0.4,
    }
    
    # Dwell time thresholds in milliseconds
    DWELL_THRESHOLDS = {
        "low": 3000,      # < 3s = probably not useful
        "medium": 10000,  # 3-10s = somewhat engaged
        "high": 30000,    # 10-30s = very engaged
        "extreme": 60000, # > 30s = extremely engaged
    }
    
    def __init__(
        self,
        signal_weights: Optional[Dict[FeedbackType, Optional[float]]] = None,
        confidence_weights: Optional[Dict[FeedbackType, float]] = None,
        min_confidence: float = 0.3,
        position_decay: float = 0.1,
    ):
        """Initialize feature extractor.
        
        Args:
            signal_weights: Custom signal weights per feedback type
            confidence_weights: Custom confidence weights per feedback type
            min_confidence: Minimum confidence threshold for features
            position_decay: Position bias decay factor
        """
        self.signal_weights = signal_weights or self.DEFAULT_SIGNAL_WEIGHTS.copy()
        self.confidence_weights = confidence_weights or self.DEFAULT_CONFIDENCE_WEIGHTS.copy()
        self.min_confidence = min_confidence
        self.position_decay = position_decay
        
        logger.info(
            f"FeatureExtractor initialized with min_confidence={min_confidence}, "
            f"position_decay={position_decay}"
        )
    
    def extract(self, feedback: EnrichedFeedback) -> Optional[TrainingFeatures]:
        """Extract training features from a single feedback event.
        
        Args:
            feedback: Enriched feedback event
            
        Returns:
            TrainingFeatures if extraction successful, None otherwise
        """
        # Validate required fields
        if not self._validate_feedback(feedback):
            logger.warning(f"Invalid feedback event: {feedback.feedback_id}")
            return None
        
        # Compute label based on feedback type
        label = self._compute_label(feedback)
        
        # Compute confidence based on feedback type and context
        confidence = self._compute_confidence(feedback)
        
        # Apply minimum confidence threshold
        if confidence < self.min_confidence:
            logger.debug(
                f"Feedback {feedback.feedback_id} below min confidence: "
                f"{confidence:.3f} < {self.min_confidence}"
            )
            return None
        
        # Determine label source
        label_source = self._determine_label_source(feedback)
        
        # Create training features
        features = TrainingFeatures(
            query_id=feedback.query_id,
            query_text=feedback.query_text,
            query_embedding=feedback.query_embedding,
            chunk_id=feedback.chunk_id,
            chunk_text=feedback.chunk_text,
            chunk_embedding=feedback.chunk_embedding,
            label=label,
            label_source=label_source,
            confidence=confidence,
            position=feedback.position_in_results or 0,
            feedback_types=[feedback.event_type.value],
            timestamp=feedback.timestamp,
            session_id=feedback.session_id,
        )
        
        logger.debug(
            f"Extracted features for {feedback.feedback_id}: "
            f"label={label:.3f}, confidence={confidence:.3f}"
        )
        
        return features
    
    def extract_batch(
        self,
        feedbacks: List[EnrichedFeedback],
        aggregate_by_pair: bool = True,
    ) -> List[TrainingFeatures]:
        """Extract training features from a batch of feedback events.
        
        Args:
            feedbacks: List of enriched feedback events
            aggregate_by_pair: If True, aggregate multiple signals for same query-doc pair
            
        Returns:
            List of training features
        """
        features_list = []
        
        # Extract individual features
        for feedback in feedbacks:
            features = self.extract(feedback)
            if features:
                features_list.append(features)
        
        # Optionally aggregate by query-document pair
        if aggregate_by_pair:
            features_list = self._aggregate_features(features_list)
        
        logger.info(
            f"Extracted {len(features_list)} features from {len(feedbacks)} feedback events"
        )
        
        return features_list
    
    def _validate_feedback(self, feedback: EnrichedFeedback) -> bool:
        """Validate feedback has required fields."""
        if not feedback.query_id or not feedback.query_text:
            return False
        if not feedback.chunk_id or not feedback.chunk_text:
            return False
        if feedback.query_embedding is None or feedback.chunk_embedding is None:
            return False
        return True
    
    def _compute_label(self, feedback: EnrichedFeedback) -> float:
        """Compute relevance label from feedback."""
        event_type = feedback.event_type
        
        # Handle rating type specially
        if event_type == FeedbackType.RATING and feedback.rating_value is not None:
            # Normalize 1-5 rating to 0-1 scale
            return (feedback.rating_value - 1) / 4.0
        
        # Handle dwell time specially
        if event_type == FeedbackType.DWELL and feedback.dwell_time_ms is not None:
            return self._compute_dwell_label(feedback.dwell_time_ms)
        
        # Use default signal weight
        weight = self.signal_weights.get(event_type)
        if weight is None:
            logger.warning(f"No signal weight for {event_type}, using 0.5")
            return 0.5
        
        return weight
    
    def _compute_dwell_label(self, dwell_time_ms: int) -> float:
        """Compute label from dwell time."""
        if dwell_time_ms < self.DWELL_THRESHOLDS["low"]:
            return 0.2
        elif dwell_time_ms < self.DWELL_THRESHOLDS["medium"]:
            return 0.5
        elif dwell_time_ms < self.DWELL_THRESHOLDS["high"]:
            return 0.75
        else:
            return 0.9
    
    def _compute_confidence(self, feedback: EnrichedFeedback) -> float:
        """Compute confidence score for the label."""
        event_type = feedback.event_type
        
        # Base confidence from feedback type
        base_confidence = self.confidence_weights.get(event_type, 0.5)
        
        # Apply position decay (lower positions = less confidence)
        if feedback.position_in_results is not None:
            position_factor = 1.0 / (1.0 + self.position_decay * feedback.position_in_results)
            base_confidence *= position_factor
        
        return min(1.0, max(0.0, base_confidence))
    
    def _determine_label_source(self, feedback: EnrichedFeedback) -> str:
        """Determine the source category of the label."""
        explicit_types = {FeedbackType.THUMBS_UP, FeedbackType.THUMBS_DOWN, FeedbackType.RATING}
        implicit_types = {FeedbackType.CLICK, FeedbackType.DWELL, FeedbackType.COPY, FeedbackType.SHARE}
        
        if feedback.event_type in explicit_types:
            return "explicit"
        elif feedback.event_type in implicit_types:
            return "implicit"
        else:
            return "inferred"
    
    def _aggregate_features(self, features_list: List[TrainingFeatures]) -> List[TrainingFeatures]:
        """Aggregate features for same query-document pairs."""
        # Group by query-chunk pair
        grouped: Dict[Tuple[str, str], List[TrainingFeatures]] = {}
        
        for features in features_list:
            key = (features.query_id, features.chunk_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(features)
        
        # Aggregate each group
        aggregated = []
        for (query_id, chunk_id), group in grouped.items():
            if len(group) == 1:
                aggregated.append(group[0])
            else:
                aggregated.append(self._merge_features(group))
        
        return aggregated
    
    def _merge_features(self, features_group: List[TrainingFeatures]) -> TrainingFeatures:
        """Merge multiple features for same query-document pair."""
        # Use first feature as base
        base = features_group[0]
        
        # Compute weighted average label
        total_weight = sum(f.confidence for f in features_group)
        weighted_label = sum(f.label * f.confidence for f in features_group) / total_weight
        
        # Combine confidence (higher with more signals)
        combined_confidence = min(1.0, max(f.confidence for f in features_group) * 1.1)
        
        # Collect all feedback types
        all_types = list(set(
            ft for f in features_group for ft in f.feedback_types
        ))
        
        # Determine merged label source
        sources = [f.label_source for f in features_group]
        if "explicit" in sources:
            label_source = "explicit"
        elif "implicit" in sources:
            label_source = "implicit"
        else:
            label_source = "inferred"
        
        # Use earliest timestamp
        earliest_timestamp = min(f.timestamp for f in features_group)
        
        return TrainingFeatures(
            query_id=base.query_id,
            query_text=base.query_text,
            query_embedding=base.query_embedding,
            chunk_id=base.chunk_id,
            chunk_text=base.chunk_text,
            chunk_embedding=base.chunk_embedding,
            label=weighted_label,
            label_source=label_source,
            confidence=combined_confidence,
            position=base.position,
            feedback_types=all_types,
            timestamp=earliest_timestamp,
            session_id=base.session_id,
        )

