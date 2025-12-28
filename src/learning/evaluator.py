"""Model evaluation for retrieval quality metrics.

This module provides comprehensive evaluation of embedding models
including ranking metrics, benchmark comparisons, and A/B test analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Evaluation metrics for retrieval quality."""
    # Core ranking metrics
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    map_score: float = 0.0  # Mean Average Precision
    
    # Recall at different k values
    recall: Dict[int, float] = field(default_factory=dict)
    
    # Precision at different k values
    precision: Dict[int, float] = field(default_factory=dict)
    
    # Additional metrics
    hit_rate: float = 0.0  # Fraction of queries with at least one relevant result
    avg_position: float = 0.0  # Average position of first relevant result
    
    # Timing metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Sample size
    num_queries: int = 0
    num_relevant: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "map": self.map_score,
            "recall": self.recall,
            "precision": self.precision,
            "hit_rate": self.hit_rate,
            "avg_position": self.avg_position,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "num_queries": self.num_queries,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two models."""
    model_a_version: str
    model_b_version: str
    
    # Metrics for each model
    model_a_metrics: EvalMetrics = field(default_factory=EvalMetrics)
    model_b_metrics: EvalMetrics = field(default_factory=EvalMetrics)
    
    # Statistical comparison
    mrr_delta: float = 0.0
    mrr_relative_improvement: float = 0.0
    ndcg_delta: float = 0.0
    is_significant: bool = False
    p_value: float = 1.0
    
    # Recommendation
    winner: str = ""  # "model_a", "model_b", or "tie"
    confidence: float = 0.0
    
    def __post_init__(self):
        """Compute deltas after initialization."""
        if self.model_a_metrics and self.model_b_metrics:
            self.mrr_delta = self.model_b_metrics.mrr - self.model_a_metrics.mrr
            if self.model_a_metrics.mrr > 0:
                self.mrr_relative_improvement = self.mrr_delta / self.model_a_metrics.mrr
            self.ndcg_delta = self.model_b_metrics.ndcg - self.model_a_metrics.ndcg


@dataclass
class BenchmarkResult:
    """Result of running a benchmark evaluation."""
    benchmark_name: str
    model_version: str
    
    # Overall metrics
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    
    # Per-query breakdown
    query_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0
    queries_per_second: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = field(default_factory=dict)


class ModelEvaluator:
    """Evaluates retrieval model quality.
    
    Computes standard IR metrics including MRR, NDCG, Recall@k, and Precision@k.
    Supports model comparison with statistical significance testing.
    """
    
    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        relevance_threshold: float = 0.5,
        significance_level: float = 0.05,
    ):
        """Initialize evaluator.
        
        Args:
            k_values: K values for recall@k and precision@k
            relevance_threshold: Threshold for binary relevance
            significance_level: P-value threshold for significance
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.relevance_threshold = relevance_threshold
        self.significance_level = significance_level
        
        logger.info(
            f"ModelEvaluator initialized with k_values={self.k_values}, "
            f"relevance_threshold={relevance_threshold}"
        )
    
    def evaluate(
        self,
        predictions: List[List[str]],
        relevance_scores: List[Dict[str, float]],
        latencies: Optional[List[float]] = None,
    ) -> EvalMetrics:
        """Evaluate predictions against relevance judgments.
        
        Args:
            predictions: List of ranked document IDs per query
            relevance_scores: List of {doc_id: relevance_score} dicts per query
            latencies: Optional list of latency measurements in ms
            
        Returns:
            EvalMetrics with computed metrics
        """
        if len(predictions) != len(relevance_scores):
            raise ValueError("predictions and relevance_scores must have same length")
        
        num_queries = len(predictions)
        
        # Compute metrics for each query
        mrr_scores = []
        ndcg_scores = []
        ap_scores = []
        recall_at_k = {k: [] for k in self.k_values}
        precision_at_k = {k: [] for k in self.k_values}
        first_positions = []
        hits = []
        
        for pred, rels in zip(predictions, relevance_scores):
            # Get binary relevance judgments
            binary_rels = {doc_id: 1 if score >= self.relevance_threshold else 0 
                          for doc_id, score in rels.items()}
            
            # Compute MRR
            mrr = self._compute_mrr(pred, binary_rels)
            mrr_scores.append(mrr)
            
            # Compute NDCG
            ndcg = self._compute_ndcg(pred, rels)
            ndcg_scores.append(ndcg)
            
            # Compute AP
            ap = self._compute_ap(pred, binary_rels)
            ap_scores.append(ap)
            
            # Compute recall and precision at k
            num_relevant = sum(binary_rels.values())
            for k in self.k_values:
                top_k = pred[:k]
                relevant_in_top_k = sum(1 for doc_id in top_k if binary_rels.get(doc_id, 0) == 1)
                
                recall = relevant_in_top_k / num_relevant if num_relevant > 0 else 0.0
                precision = relevant_in_top_k / k if k > 0 else 0.0
                
                recall_at_k[k].append(recall)
                precision_at_k[k].append(precision)
            
            # Track first relevant position
            first_pos = self._find_first_relevant(pred, binary_rels)
            if first_pos is not None:
                first_positions.append(first_pos)
                hits.append(1)
            else:
                hits.append(0)
        
        # Aggregate metrics
        metrics = EvalMetrics(
            mrr=np.mean(mrr_scores),
            ndcg=np.mean(ndcg_scores),
            map_score=np.mean(ap_scores),
            recall={k: np.mean(values) for k, values in recall_at_k.items()},
            precision={k: np.mean(values) for k, values in precision_at_k.items()},
            hit_rate=np.mean(hits),
            avg_position=np.mean(first_positions) if first_positions else 0.0,
            num_queries=num_queries,
            num_relevant=sum(sum(1 for s in r.values() if s >= self.relevance_threshold) 
                           for r in relevance_scores),
        )
        
        # Add latency metrics if provided
        if latencies:
            metrics.avg_latency_ms = np.mean(latencies)
            metrics.p50_latency_ms = np.percentile(latencies, 50)
            metrics.p95_latency_ms = np.percentile(latencies, 95)
            metrics.p99_latency_ms = np.percentile(latencies, 99)
        
        return metrics
    
    def compare_models(
        self,
        model_a_predictions: List[List[str]],
        model_b_predictions: List[List[str]],
        relevance_scores: List[Dict[str, float]],
        model_a_version: str = "baseline",
        model_b_version: str = "candidate",
    ) -> ComparisonResult:
        """Compare two models statistically.
        
        Args:
            model_a_predictions: Predictions from model A
            model_b_predictions: Predictions from model B
            relevance_scores: Ground truth relevance scores
            model_a_version: Version string for model A
            model_b_version: Version string for model B
            
        Returns:
            ComparisonResult with statistical analysis
        """
        # Evaluate both models
        metrics_a = self.evaluate(model_a_predictions, relevance_scores)
        metrics_b = self.evaluate(model_b_predictions, relevance_scores)
        
        # Compute per-query MRR for paired test
        mrr_a = []
        mrr_b = []
        
        for pred_a, pred_b, rels in zip(model_a_predictions, model_b_predictions, relevance_scores):
            binary_rels = {doc_id: 1 if score >= self.relevance_threshold else 0 
                          for doc_id, score in rels.items()}
            mrr_a.append(self._compute_mrr(pred_a, binary_rels))
            mrr_b.append(self._compute_mrr(pred_b, binary_rels))
        
        # Paired t-test for significance
        p_value, is_significant = self._paired_ttest(mrr_a, mrr_b)
        
        # Determine winner
        if is_significant:
            if metrics_b.mrr > metrics_a.mrr:
                winner = "model_b"
                confidence = 1.0 - p_value
            else:
                winner = "model_a"
                confidence = 1.0 - p_value
        else:
            winner = "tie"
            confidence = 0.5
        
        result = ComparisonResult(
            model_a_version=model_a_version,
            model_b_version=model_b_version,
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            is_significant=is_significant,
            p_value=p_value,
            winner=winner,
            confidence=confidence,
        )
        
        return result
    
    def run_benchmark(
        self,
        model: Any,
        queries: List[str],
        ground_truth: List[Dict[str, float]],
        model_version: str,
        benchmark_name: str = "default",
        top_k: int = 100,
    ) -> BenchmarkResult:
        """Run benchmark evaluation on a model.
        
        Args:
            model: Model with encode() and search() methods
            queries: List of query strings
            ground_truth: List of relevance dictionaries
            model_version: Version of the model
            benchmark_name: Name of the benchmark
            top_k: Number of results to retrieve
            
        Returns:
            BenchmarkResult with detailed metrics
        """
        predictions = []
        latencies = []
        query_metrics = []
        
        start_time = time.time()
        
        for i, query in enumerate(queries):
            # Time the query
            query_start = time.time()
            
            # Get predictions from model
            results = self._run_query(model, query, top_k)
            
            query_latency = (time.time() - query_start) * 1000
            latencies.append(query_latency)
            predictions.append(results)
            
            # Compute per-query metrics
            if ground_truth[i]:
                binary_rels = {doc_id: 1 if score >= self.relevance_threshold else 0 
                              for doc_id, score in ground_truth[i].items()}
                query_metrics.append({
                    "query_idx": i,
                    "mrr": self._compute_mrr(results, binary_rels),
                    "latency_ms": query_latency,
                    "num_results": len(results),
                })
        
        total_time = time.time() - start_time
        
        # Compute overall metrics
        metrics = self.evaluate(predictions, ground_truth, latencies)
        
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            model_version=model_version,
            metrics=metrics,
            query_metrics=query_metrics,
            total_time_seconds=total_time,
            queries_per_second=len(queries) / total_time if total_time > 0 else 0,
            config={
                "top_k": top_k,
                "num_queries": len(queries),
            },
        )
        
        logger.info(
            f"Benchmark {benchmark_name} completed: "
            f"MRR={metrics.mrr:.4f}, NDCG={metrics.ndcg:.4f}, "
            f"QPS={result.queries_per_second:.2f}"
        )
        
        return result
    
    def _compute_mrr(self, predictions: List[str], binary_rels: Dict[str, int]) -> float:
        """Compute Mean Reciprocal Rank for a single query."""
        for i, doc_id in enumerate(predictions):
            if binary_rels.get(doc_id, 0) == 1:
                return 1.0 / (i + 1)
        return 0.0
    
    def _compute_ndcg(
        self,
        predictions: List[str],
        relevance: Dict[str, float],
        k: Optional[int] = None,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        if k is None:
            k = len(predictions)
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(predictions[:k]):
            rel = relevance.get(doc_id, 0.0)
            dcg += (2**rel - 1) / np.log2(i + 2)
        
        # Ideal DCG
        ideal_rels = sorted(relevance.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2**rel - 1) / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _compute_ap(self, predictions: List[str], binary_rels: Dict[str, int]) -> float:
        """Compute Average Precision for a single query."""
        num_relevant = sum(binary_rels.values())
        if num_relevant == 0:
            return 0.0
        
        precision_sum = 0.0
        num_hits = 0
        
        for i, doc_id in enumerate(predictions):
            if binary_rels.get(doc_id, 0) == 1:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / num_relevant
    
    def _find_first_relevant(
        self,
        predictions: List[str],
        binary_rels: Dict[str, int],
    ) -> Optional[int]:
        """Find position of first relevant result (1-indexed)."""
        for i, doc_id in enumerate(predictions):
            if binary_rels.get(doc_id, 0) == 1:
                return i + 1
        return None
    
    def _paired_ttest(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> Tuple[float, bool]:
        """Perform paired t-test."""
        from scipy import stats
        
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return 1.0, False
        
        try:
            statistic, p_value = stats.ttest_rel(scores_a, scores_b)
            is_significant = p_value < self.significance_level
            return p_value, is_significant
        except Exception as e:
            logger.warning(f"t-test failed: {e}")
            return 1.0, False
    
    def _run_query(self, model: Any, query: str, top_k: int) -> List[str]:
        """Run a query through the model."""
        if hasattr(model, "search"):
            results = model.search(query, top_k=top_k)
            if isinstance(results, list):
                if results and isinstance(results[0], tuple):
                    return [r[0] for r in results]
                return results
        elif hasattr(model, "encode"):
            embedding = model.encode(query)
            if hasattr(model, "index"):
                return model.index.search(embedding, top_k)
        
        logger.warning("Model does not have expected interface")
        return []

