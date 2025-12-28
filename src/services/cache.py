"""Caching service for RAG system.

This module provides caching functionality using Redis to improve
response times and reduce load on backend services.
"""

import json
import logging
import hashlib
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, asdict

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from src.config import RedisConfig, get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached entry with metadata."""
    key: str
    value: Any
    ttl: int
    created_at: float


class CacheService:
    """Redis-based caching service.

    Provides caching for:
    - Query results
    - Embeddings
    - Document metadata
    """

    # Cache key prefixes
    PREFIX_QUERY = "query:"
    PREFIX_EMBEDDING = "embedding:"
    PREFIX_DOCUMENT = "document:"
    PREFIX_RESPONSE = "response:"

    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        client: Optional["redis.Redis"] = None,
    ):
        """Initialize cache service.

        Args:
            config: Redis configuration
            client: Pre-initialized Redis client
        """
        self.config = config or get_settings().redis
        self._client = client
        self._initialized = client is not None

    async def _ensure_initialized(self):
        """Lazy initialization of Redis client."""
        if not self._initialized:
            if not HAS_REDIS:
                raise ImportError(
                    "redis package required. Install with: pip install redis"
                )
            logger.info(f"Connecting to Redis: {self.config.url}")
            self._client = redis.from_url(self.config.url)
            self._initialized = True

    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate a cache key."""
        return f"{prefix}{identifier}"

    def _hash_query(self, query: str, filters: Optional[Dict] = None) -> str:
        """Generate a hash for a query."""
        content = query
        if filters:
            content += json.dumps(filters, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        await self._ensure_initialized()

        try:
            value = await self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            Success status
        """
        await self._ensure_initialized()

        ttl = ttl or self.config.cache_ttl

        try:
            serialized = json.dumps(value)
            await self._client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            Success status
        """
        await self._ensure_initialized()

        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "query:*")

        Returns:
            Number of deleted keys
        """
        await self._ensure_initialized()

        try:
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self._client.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.warning(f"Cache delete pattern error: {e}")
            return 0

    # Query caching
    async def get_query_result(
        self,
        query: str,
        filters: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Get cached query result.

        Args:
            query: Query string
            filters: Query filters

        Returns:
            Cached result or None
        """
        key = self._generate_key(
            self.PREFIX_QUERY,
            self._hash_query(query, filters),
        )
        return await self.get(key)

    async def set_query_result(
        self,
        query: str,
        result: Dict,
        filters: Optional[Dict] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a query result.

        Args:
            query: Query string
            result: Query result to cache
            filters: Query filters
            ttl: Time-to-live

        Returns:
            Success status
        """
        key = self._generate_key(
            self.PREFIX_QUERY,
            self._hash_query(query, filters),
        )
        return await self.set(key, result, ttl)

    # Embedding caching
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding.

        Args:
            text: Text that was embedded

        Returns:
            Embedding vector or None
        """
        key = self._generate_key(
            self.PREFIX_EMBEDDING,
            hashlib.md5(text.encode()).hexdigest(),
        )
        return await self.get(key)

    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache an embedding.

        Args:
            text: Original text
            embedding: Embedding vector
            ttl: Time-to-live

        Returns:
            Success status
        """
        key = self._generate_key(
            self.PREFIX_EMBEDDING,
            hashlib.md5(text.encode()).hexdigest(),
        )
        # Cache embeddings for longer (24 hours default)
        ttl = ttl or 86400
        return await self.set(key, embedding, ttl)

    # Response caching
    async def get_response(self, response_id: str) -> Optional[Dict]:
        """Get cached response by ID.

        Args:
            response_id: Response identifier

        Returns:
            Cached response or None
        """
        key = self._generate_key(self.PREFIX_RESPONSE, response_id)
        return await self.get(key)

    async def set_response(
        self,
        response_id: str,
        response: Dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a response.

        Args:
            response_id: Response identifier
            response: Response data
            ttl: Time-to-live

        Returns:
            Success status
        """
        key = self._generate_key(self.PREFIX_RESPONSE, response_id)
        return await self.set(key, response, ttl)

    # Document caching
    async def invalidate_document(self, document_id: str) -> int:
        """Invalidate all cache entries for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of invalidated entries
        """
        # Invalidate document-specific entries
        count = await self.delete_pattern(f"{self.PREFIX_DOCUMENT}{document_id}*")

        # Also invalidate related query cache
        # (In production, would use more sophisticated invalidation)
        count += await self.delete_pattern(f"{self.PREFIX_QUERY}*")

        return count

    async def health_check(self) -> bool:
        """Check cache health.

        Returns:
            True if healthy
        """
        try:
            await self._ensure_initialized()
            await self._client.ping()
            return True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False

    async def close(self):
        """Close the cache connection."""
        if self._client:
            await self._client.close()
            self._initialized = False


# Singleton instance
_cache: Optional[CacheService] = None


def get_cache() -> CacheService:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache
