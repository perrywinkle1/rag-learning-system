"""Re-ranking service for improving retrieval quality.

This module provides cross-encoder based re-ranking to refine
initial retrieval results.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

from src.config import RerankerConfig, get_settings
from src.core.retrieval import RetrievalCandidate

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """A re-ranked candidate with additional scoring information."""
    chunk_id: str
    document_id: str
    content: str
    original_score: float
    reranked_score: float
    rank: int
    metadata: Dict[str, Any]

    @property
    def final_score(self) -> float:
        """Get the final score (reranked if available)."""
        return self.reranked_score


class Reranker:
    """Service for re-ranking retrieval candidates.

    Uses a cross-encoder model to compute more accurate relevance
    scores between queries and documents.
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        model: Optional["CrossEncoder"] = None,
    ):
        """Initialize the reranker.

        Args:
            config: Reranker configuration
            model: Pre-loaded model (for testing)
        """
        self.config = config or get_settings().reranker
        self._model = model
        self._initialized = model is not None

    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if not self._initialized:
            if not HAS_CROSS_ENCODER:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"Loading reranker model: {self.config.model_name}")
            self._model = CrossEncoder(self.config.model_name)
            self._initialized = True
            logger.info("Reranker model loaded")

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: Optional[int] = None,
    ) -> List[RankedCandidate]:
        """Re-rank candidates using cross-encoder.

        Args:
            query: Search query
            candidates: Initial retrieval candidates
            top_k: Number of results to return

        Returns:
            Re-ranked candidates
        """
        if not self.config.enabled:
            # Return original order if reranking disabled
            return [
                RankedCandidate(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    original_score=c.score,
                    reranked_score=c.score,
                    rank=i + 1,
                    metadata=c.metadata,
                )
                for i, c in enumerate(candidates)
            ]

        self._ensure_initialized()

        top_k = top_k or self.config.top_k

        # Limit candidates for efficiency
        candidates_to_rerank = candidates[:self.config.max_candidates]

        if not candidates_to_rerank:
            return []

        # Create query-document pairs
        pairs = [(query, c.content) for c in candidates_to_rerank]

        # Get cross-encoder scores
        scores = self._model.predict(pairs)

        # Combine with candidates
        scored_candidates = list(zip(candidates_to_rerank, scores))

        # Sort by reranked score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Create ranked candidates
        ranked = []
        for i, (candidate, score) in enumerate(scored_candidates[:top_k]):
            ranked.append(RankedCandidate(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                content=candidate.content,
                original_score=candidate.score,
                reranked_score=float(score),
                rank=i + 1,
                metadata=candidate.metadata,
            ))

        return ranked

    def compute_features(
        self,
        query: str,
        candidate: RetrievalCandidate,
    ) -> Dict[str, float]:
        """Compute feature scores for a candidate.

        Useful for understanding why a document was ranked highly.

        Args:
            query: Search query
            candidate: Candidate document

        Returns:
            Dictionary of feature scores
        """
        self._ensure_initialized()

        # Cross-encoder score
        ce_score = float(self._model.predict([(query, candidate.content)])[0])

        # Additional features could be computed here
        features = {
            "cross_encoder_score": ce_score,
            "original_score": candidate.score,
            "content_length": len(candidate.content),
        }

        return features


class HybridReranker(Reranker):
    """Reranker that combines multiple scoring signals.

    Combines cross-encoder scores with other features like
    recency, document quality, etc.
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        model: Optional["CrossEncoder"] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize hybrid reranker.

        Args:
            config: Reranker configuration
            model: Pre-loaded model
            weights: Feature weights for combining scores
        """
        super().__init__(config, model)
        self.weights = weights or {
            "cross_encoder": 0.7,
            "original": 0.2,
            "recency": 0.1,
        }

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: Optional[int] = None,
    ) -> List[RankedCandidate]:
        """Re-rank using hybrid scoring.

        Args:
            query: Search query
            candidates: Initial candidates
            top_k: Number of results

        Returns:
            Re-ranked candidates
        """
        if not self.config.enabled or not candidates:
            return await super().rerank(query, candidates, top_k)

        self._ensure_initialized()

        top_k = top_k or self.config.top_k
        candidates_to_rerank = candidates[:self.config.max_candidates]

        # Get cross-encoder scores
        pairs = [(query, c.content) for c in candidates_to_rerank]
        ce_scores = self._model.predict(pairs)

        # Normalize cross-encoder scores to 0-1
        if len(ce_scores) > 1:
            ce_min, ce_max = min(ce_scores), max(ce_scores)
            if ce_max > ce_min:
                ce_scores_norm = [(s - ce_min) / (ce_max - ce_min) for s in ce_scores]
            else:
                ce_scores_norm = [0.5] * len(ce_scores)
        else:
            ce_scores_norm = [0.5] * len(ce_scores)

        # Compute hybrid scores
        scored_candidates = []
        for i, candidate in enumerate(candidates_to_rerank):
            # Get recency score (placeholder - would use actual timestamps)
            recency_score = 1.0 - (i / len(candidates_to_rerank)) * 0.2

            # Combine scores
            hybrid_score = (
                self.weights["cross_encoder"] * ce_scores_norm[i] +
                self.weights["original"] * candidate.score +
                self.weights["recency"] * recency_score
            )

            scored_candidates.append((candidate, float(ce_scores[i]), hybrid_score))

        # Sort by hybrid score
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        # Create ranked candidates
        ranked = []
        for i, (candidate, ce_score, hybrid_score) in enumerate(scored_candidates[:top_k]):
            ranked.append(RankedCandidate(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                content=candidate.content,
                original_score=candidate.score,
                reranked_score=hybrid_score,
                rank=i + 1,
                metadata=candidate.metadata,
            ))

        return ranked


# Singleton instance
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """Get the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
