"""Configuration management for RAG Learning System."""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from functools import lru_cache


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "postgresql://rag:rag_password@postgres:5432/rag_learning"
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class QdrantConfig:
    """Qdrant vector store configuration."""
    url: str = "http://qdrant:6333"
    collection_name: str = "documents"
    vector_size: int = 384  # all-MiniLM-L6-v2 dimensions


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: str = "kafka:9092"
    feedback_topic: str = "feedback-events"
    consumer_group: str = "rag-learning"


@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str = "redis://redis:6379"
    cache_ttl: int = 3600  # 1 hour


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    max_tokens: int = 512
    batch_size: int = 32
    pooling_strategy: str = "mean"


@dataclass
class RerankerConfig:
    """Re-ranker configuration."""
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_candidates: int = 50
    top_k: int = 10


@dataclass
class GenerationConfig:
    """LLM generation configuration."""
    provider: str = "openai"  # openai, anthropic
    model: str = "gpt-4-turbo"
    max_tokens: int = 1024
    temperature: float = 0.3

    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k_candidates: int = 100
    top_k_final: int = 10
    min_score: float = 0.5


@dataclass
class LearningConfig:
    """Learning pipeline configuration."""
    enabled: bool = True
    min_feedback_for_training: int = 1000
    training_schedule: str = "0 2 * * *"  # Daily at 2 AM
    validation_threshold: float = 0.95
    auto_rollback: bool = True


@dataclass
class ABTestingConfig:
    """A/B testing configuration."""
    enabled: bool = True
    traffic_split: float = 0.1  # 10% to challenger
    min_samples: int = 500
    significance_level: float = 0.05


@dataclass
class Settings:
    """Main application settings."""
    app_name: str = "RAG Learning System"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)


@lru_cache()
def get_settings() -> Settings:
    """Get application settings from environment."""
    return Settings(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        database=DatabaseConfig(
            url=os.getenv("DATABASE_URL", DatabaseConfig.url),
        ),
        qdrant=QdrantConfig(
            url=os.getenv("QDRANT_URL", QdrantConfig.url),
        ),
        kafka=KafkaConfig(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", KafkaConfig.bootstrap_servers),
        ),
        redis=RedisConfig(
            url=os.getenv("REDIS_URL", RedisConfig.url),
        ),
        generation=GenerationConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        ),
    )
