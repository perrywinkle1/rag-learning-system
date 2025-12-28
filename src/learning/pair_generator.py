"""Training pair generation for contrastive learning.

This module generates positive/negative pairs from training features
for contrastive learning objectives.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import numpy as np

from src.learning.feature_extractor import TrainingFeatures

logger = logging.getLogger(__name__)


class PairType(Enum):
    """Types of training pairs."""
    POSITIVE_NEGATIVE = "positive_negative"
    TRIPLET = "triplet"
    HARD_NEGATIVE = "hard_negative"


@dataclass
class TrainingPair:
    """A training pair for contrastive learning."""
    pair_id: str
    pair_type: PairType
    
    # Query information
    query_id: str
    query_text: str
    query_embedding: np.ndarray
    
    # Positive document (anchor for triplet)
    positive_chunk_id: str
    positive_chunk_text: str
    positive_chunk_embedding: np.ndarray
    positive_score: float
    
    # Negative document
    negative_chunk_id: str
    negative_chunk_text: str
    negative_chunk_embedding: np.ndarray
    negative_score: float
    
    # Metadata
    label_source: str
    confidence: float
    is_hard_negative: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TripletSample:
    """A triplet sample (anchor, positive, negative) for triplet loss."""
    anchor_embedding: np.ndarray
    positive_embedding: np.ndarray
    negative_embedding: np.ndarray
    
    anchor_text: str
    positive_text: str
    negative_text: str
    
    margin: float = 0.0


class TrainingPairGenerator:
    """Generates training pairs from feedback-derived features.
    
    Supports multiple pair generation strategies:
    - Random negatives: Random documents not clicked for the query
    - In-batch negatives: Other positives in the batch as negatives
    - Hard negatives: High-ranked but not clicked documents
    """
    
    def __init__(
        self,
        positive_threshold: float = 0.6,
        negative_threshold: float = 0.4,
        hard_negative_rank_threshold: int = 10,
        min_confidence: float = 0.5,
        max_negatives_per_positive: int = 5,
        hard_negative_ratio: float = 0.3,
        random_seed: Optional[int] = None,
    ):
        """Initialize pair generator.
        
        Args:
            positive_threshold: Minimum label for positive samples
            negative_threshold: Maximum label for negative samples
            hard_negative_rank_threshold: Max rank for hard negatives
            min_confidence: Minimum confidence for pair inclusion
            max_negatives_per_positive: Max negatives per positive sample
            hard_negative_ratio: Fraction of hard negatives vs random
            random_seed: Random seed for reproducibility
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.hard_negative_rank_threshold = hard_negative_rank_threshold
        self.min_confidence = min_confidence
        self.max_negatives_per_positive = max_negatives_per_positive
        self.hard_negative_ratio = hard_negative_ratio
        
        self.rng = random.Random(random_seed)
        
        logger.info(
            f"TrainingPairGenerator initialized: "
            f"pos_threshold={positive_threshold}, neg_threshold={negative_threshold}, "
            f"hard_neg_ratio={hard_negative_ratio}"
        )
    
    def generate_pairs(
        self,
        features: List[TrainingFeatures],
        use_hard_negatives: bool = True,
        use_in_batch_negatives: bool = True,
    ) -> List[TrainingPair]:
        """Generate training pairs from features.
        
        Args:
            features: List of training features
            use_hard_negatives: Whether to include hard negatives
            use_in_batch_negatives: Whether to use in-batch negatives
            
        Returns:
            List of training pairs
        """
        # Separate positives and negatives
        positives, negatives = self._partition_features(features)
        
        if not positives:
            logger.warning("No positive samples found")
            return []
        
        if not negatives and not use_in_batch_negatives:
            logger.warning("No negative samples and in-batch negatives disabled")
            return []
        
        pairs = []
        
        # Group features by query
        query_features = self._group_by_query(features)
        
        for query_id, query_feats in query_features.items():
            query_positives = [f for f in query_feats if f.label >= self.positive_threshold]
            query_negatives = [f for f in query_feats if f.label <= self.negative_threshold]
            
            if not query_positives:
                continue
            
            # Identify hard negatives (high position but low label)
            hard_negatives = []
            random_negatives = []
            
            if use_hard_negatives:
                for neg in query_negatives:
                    if neg.position <= self.hard_negative_rank_threshold:
                        hard_negatives.append(neg)
                    else:
                        random_negatives.append(neg)
            else:
                random_negatives = query_negatives
            
            # Generate pairs for each positive
            for pos in query_positives:
                # Determine number of hard vs random negatives
                num_negatives = min(
                    self.max_negatives_per_positive,
                    len(hard_negatives) + len(random_negatives)
                )
                
                if num_negatives == 0 and use_in_batch_negatives:
                    # Use other queries positives as negatives
                    other_positives = [
                        f for qid, qf in query_features.items()
                        if qid != query_id
                        for f in qf if f.label >= self.positive_threshold
                    ]
                    if other_positives:
                        selected_negs = self.rng.sample(
                            other_positives,
                            min(self.max_negatives_per_positive, len(other_positives))
                        )
                        for neg in selected_negs:
                            pair = self._create_pair(pos, neg, is_hard=False)
                            pairs.append(pair)
                    continue
                
                num_hard = int(num_negatives * self.hard_negative_ratio)
                num_random = num_negatives - num_hard
                
                # Select hard negatives
                selected_hard = self.rng.sample(
                    hard_negatives,
                    min(num_hard, len(hard_negatives))
                )
                
                # Select random negatives
                selected_random = self.rng.sample(
                    random_negatives,
                    min(num_random, len(random_negatives))
                )
                
                # Create pairs
                for neg in selected_hard:
                    pair = self._create_pair(pos, neg, is_hard=True)
                    pairs.append(pair)
                
                for neg in selected_random:
                    pair = self._create_pair(pos, neg, is_hard=False)
                    pairs.append(pair)
        
        logger.info(
            f"Generated {len(pairs)} pairs from {len(features)} features "
            f"({len(positives)} positives, {len(negatives)} negatives)"
        )
        
        return pairs
    
    def generate_triplets(
        self,
        features: List[TrainingFeatures],
        margin: float = 0.2,
    ) -> List[TripletSample]:
        """Generate triplet samples for triplet loss.
        
        Args:
            features: List of training features
            margin: Desired margin between positive and negative
            
        Returns:
            List of triplet samples
        """
        pairs = self.generate_pairs(features)
        
        triplets = []
        for pair in pairs:
            triplet = TripletSample(
                anchor_embedding=pair.query_embedding,
                positive_embedding=pair.positive_chunk_embedding,
                negative_embedding=pair.negative_chunk_embedding,
                anchor_text=pair.query_text,
                positive_text=pair.positive_chunk_text,
                negative_text=pair.negative_chunk_text,
                margin=margin,
            )
            triplets.append(triplet)
        
        return triplets
    
    def mine_hard_negatives(
        self,
        query_embedding: np.ndarray,
        positive_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidate_ids: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, np.ndarray, float]]:
        """Mine hard negatives using embedding similarity.
        
        Hard negatives are candidates that are similar to the positive
        but should be ranked lower.
        
        Args:
            query_embedding: Query embedding
            positive_embedding: Positive document embedding
            candidate_embeddings: Pool of candidate embeddings
            candidate_ids: IDs corresponding to candidates
            top_k: Number of hard negatives to return
            
        Returns:
            List of (id, embedding, similarity) tuples
        """
        if not candidate_embeddings:
            return []
        
        # Compute similarities to query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        similarities = []
        for i, cand_emb in enumerate(candidate_embeddings):
            cand_norm = cand_emb / np.linalg.norm(cand_emb)
            sim = float(np.dot(query_norm, cand_norm))
            similarities.append((candidate_ids[i], cand_emb, sim))
        
        # Sort by similarity (descending) and take top-k
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities[:top_k]
    
    def _partition_features(
        self,
        features: List[TrainingFeatures],
    ) -> Tuple[List[TrainingFeatures], List[TrainingFeatures]]:
        """Partition features into positives and negatives."""
        positives = []
        negatives = []
        
        for f in features:
            if f.confidence < self.min_confidence:
                continue
            
            if f.label >= self.positive_threshold:
                positives.append(f)
            elif f.label <= self.negative_threshold:
                negatives.append(f)
        
        return positives, negatives
    
    def _group_by_query(
        self,
        features: List[TrainingFeatures],
    ) -> Dict[str, List[TrainingFeatures]]:
        """Group features by query ID."""
        grouped: Dict[str, List[TrainingFeatures]] = {}
        
        for f in features:
            if f.query_id not in grouped:
                grouped[f.query_id] = []
            grouped[f.query_id].append(f)
        
        return grouped
    
    def _create_pair(
        self,
        positive: TrainingFeatures,
        negative: TrainingFeatures,
        is_hard: bool,
    ) -> TrainingPair:
        """Create a training pair from positive and negative features."""
        import uuid
        
        pair_type = PairType.HARD_NEGATIVE if is_hard else PairType.POSITIVE_NEGATIVE
        
        # Combined confidence
        confidence = min(positive.confidence, negative.confidence)
        
        # Use explicit source if available
        if positive.label_source == "explicit" or negative.label_source == "explicit":
            label_source = "explicit"
        elif positive.label_source == "implicit" or negative.label_source == "implicit":
            label_source = "implicit"
        else:
            label_source = "inferred"
        
        return TrainingPair(
            pair_id=str(uuid.uuid4()),
            pair_type=pair_type,
            query_id=positive.query_id,
            query_text=positive.query_text,
            query_embedding=positive.query_embedding,
            positive_chunk_id=positive.chunk_id,
            positive_chunk_text=positive.chunk_text,
            positive_chunk_embedding=positive.chunk_embedding,
            positive_score=positive.label,
            negative_chunk_id=negative.chunk_id,
            negative_chunk_text=negative.chunk_text,
            negative_chunk_embedding=negative.chunk_embedding,
            negative_score=negative.label,
            label_source=label_source,
            confidence=confidence,
            is_hard_negative=is_hard,
        )

