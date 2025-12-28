"""Embedding service for text vectorization.

This module provides functionality to convert text into dense vector
representations using sentence transformer models.
"""

import logging
from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.config import EmbeddingConfig, get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """Service for generating text embeddings.

    Uses sentence-transformers models to convert text into dense vectors
    for similarity search.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        model: Optional["SentenceTransformer"] = None,
    ):
        """Initialize the embedder.

        Args:
            config: Embedding configuration
            model: Pre-loaded model (for testing)
        """
        self.config = config or get_settings().embedding
        self._model = model
        self._initialized = model is not None

    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if not self._initialized:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
            self._initialized = True
            logger.info(f"Embedding model loaded. Dimensions: {self.config.dimensions}")

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of shape (dimensions,)
        """
        self._ensure_initialized()

        # Truncate if too long
        if len(text.split()) > self.config.max_tokens:
            tokens = text.split()[:self.config.max_tokens]
            text = " ".join(tokens)

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            Numpy array of shape (n_texts, dimensions)
        """
        self._ensure_initialized()

        # Truncate texts if needed
        processed_texts = []
        for text in texts:
            if len(text.split()) > self.config.max_tokens:
                tokens = text.split()[:self.config.max_tokens]
                text = " ".join(tokens)
            processed_texts.append(text)

        embeddings = self._model.encode(
            processed_texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query.

        This is an alias for embed() but can be extended for
        query-specific processing.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        return self.embed(query)

    def embed_document(self, document: str) -> np.ndarray:
        """Generate embedding for a document.

        This is an alias for embed() but can be extended for
        document-specific processing.

        Args:
            document: Document text

        Returns:
            Document embedding
        """
        return self.embed(document)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.config.dimensions

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0-1 for normalized vectors)
        """
        # For normalized vectors, cosine similarity = dot product
        return float(np.dot(embedding1, embedding2))


# Singleton instance
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
