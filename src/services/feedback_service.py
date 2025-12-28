"""Feedback collection and processing service.

This module provides functionality to collect and process user feedback
for the learning pipeline.
"""

import logging
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

try:
    from aiokafka import AIOKafkaProducer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

from src.config import KafkaConfig, get_settings

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback events."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CLICK = "click"
    DWELL = "dwell"
    COPY = "copy"
    SHARE = "share"


class FeedbackReason(str, Enum):
    """Reasons for negative feedback."""
    WRONG_ANSWER = "wrong_answer"
    OUTDATED = "outdated"
    IRRELEVANT = "irrelevant"
    INCOMPLETE = "incomplete"
    OTHER = "other"


@dataclass
class FeedbackEvent:
    """A feedback event from the user."""
    feedback_id: str
    response_id: str
    session_id: Optional[str]
    feedback_type: FeedbackType
    value: Optional[float] = None  # For ratings (1-5) or dwell time (ms)
    target_chunk_id: Optional[str] = None
    reason_code: Optional[FeedbackReason] = None
    reason_text: Optional[str] = None
    client_timestamp: Optional[datetime] = None
    server_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feedback_id": self.feedback_id,
            "response_id": self.response_id,
            "session_id": self.session_id,
            "feedback_type": self.feedback_type.value,
            "value": self.value,
            "target_chunk_id": self.target_chunk_id,
            "reason_code": self.reason_code.value if self.reason_code else None,
            "reason_text": self.reason_text,
            "client_timestamp": self.client_timestamp.isoformat() if self.client_timestamp else None,
            "server_timestamp": self.server_timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FeedbackResult:
    """Result of submitting feedback."""
    feedback_id: str
    accepted: bool
    message: str = ""
    error: Optional[str] = None


class FeedbackService:
    """Service for collecting and processing feedback.

    Collects feedback events and publishes them to Kafka for
    processing by the learning pipeline.
    """

    # Validation constraints
    MIN_RATING = 1
    MAX_RATING = 5
    MAX_DWELL_TIME_MS = 3600000  # 1 hour
    MAX_REASON_LENGTH = 1000

    def __init__(
        self,
        config: Optional[KafkaConfig] = None,
        producer: Optional["AIOKafkaProducer"] = None,
    ):
        """Initialize the feedback service.

        Args:
            config: Kafka configuration
            producer: Pre-initialized Kafka producer
        """
        self.config = config or get_settings().kafka
        self._producer = producer
        self._initialized = producer is not None

        # In-memory buffer for when Kafka is unavailable
        self._buffer: List[FeedbackEvent] = []
        self._buffer_limit = 10000

    async def _ensure_initialized(self):
        """Lazy initialization of Kafka producer."""
        if not self._initialized:
            if not HAS_KAFKA:
                logger.warning("Kafka not available, using in-memory buffer")
                return

            try:
                logger.info(f"Connecting to Kafka: {self.config.bootstrap_servers}")
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.config.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                )
                await self._producer.start()
                self._initialized = True
                logger.info("Kafka producer initialized")
            except Exception as e:
                logger.error(f"Failed to connect to Kafka: {e}")

    def _validate_feedback(self, event: FeedbackEvent) -> Optional[str]:
        """Validate a feedback event.

        Returns:
            Error message if invalid, None if valid
        """
        # Validate rating value
        if event.feedback_type == FeedbackType.RATING:
            if event.value is None:
                return "Rating value is required"
            if not (self.MIN_RATING <= event.value <= self.MAX_RATING):
                return f"Rating must be between {self.MIN_RATING} and {self.MAX_RATING}"

        # Validate dwell time
        if event.feedback_type == FeedbackType.DWELL:
            if event.value is None:
                return "Dwell time value is required"
            if event.value < 0 or event.value > self.MAX_DWELL_TIME_MS:
                return f"Dwell time must be between 0 and {self.MAX_DWELL_TIME_MS}ms"

        # Validate reason text
        if event.reason_text and len(event.reason_text) > self.MAX_REASON_LENGTH:
            return f"Reason text must be less than {self.MAX_REASON_LENGTH} characters"

        return None

    async def submit_feedback(
        self,
        response_id: str,
        feedback_type: FeedbackType,
        session_id: Optional[str] = None,
        value: Optional[float] = None,
        target_chunk_id: Optional[str] = None,
        reason_code: Optional[FeedbackReason] = None,
        reason_text: Optional[str] = None,
        client_timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackResult:
        """Submit a feedback event.

        Args:
            response_id: ID of the response being rated
            feedback_type: Type of feedback
            session_id: User session ID
            value: Numeric value (for ratings/dwell)
            target_chunk_id: Specific chunk being rated
            reason_code: Reason for negative feedback
            reason_text: Free-form reason text
            client_timestamp: When feedback was given on client
            metadata: Additional metadata

        Returns:
            FeedbackResult indicating success/failure
        """
        feedback_id = str(uuid.uuid4())

        event = FeedbackEvent(
            feedback_id=feedback_id,
            response_id=response_id,
            session_id=session_id,
            feedback_type=feedback_type,
            value=value,
            target_chunk_id=target_chunk_id,
            reason_code=reason_code,
            reason_text=reason_text,
            client_timestamp=client_timestamp,
            metadata=metadata or {},
        )

        # Validate
        error = self._validate_feedback(event)
        if error:
            return FeedbackResult(
                feedback_id=feedback_id,
                accepted=False,
                error=error,
            )

        # Publish to Kafka or buffer
        await self._ensure_initialized()

        try:
            if self._producer and self._initialized:
                await self._producer.send_and_wait(
                    self.config.feedback_topic,
                    event.to_dict(),
                )
                logger.debug(f"Feedback published: {feedback_id}")
            else:
                # Buffer locally
                if len(self._buffer) < self._buffer_limit:
                    self._buffer.append(event)
                    logger.debug(f"Feedback buffered: {feedback_id}")
                else:
                    logger.warning("Feedback buffer full, dropping event")

            return FeedbackResult(
                feedback_id=feedback_id,
                accepted=True,
                message="Feedback recorded",
            )

        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return FeedbackResult(
                feedback_id=feedback_id,
                accepted=False,
                error=str(e),
            )

    async def submit_batch(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Submit multiple feedback events.

        Args:
            events: List of feedback event dictionaries

        Returns:
            Summary of results
        """
        accepted = 0
        rejected = 0
        errors = []

        for event_dict in events:
            try:
                result = await self.submit_feedback(
                    response_id=event_dict["response_id"],
                    feedback_type=FeedbackType(event_dict["type"]),
                    session_id=event_dict.get("session_id"),
                    value=event_dict.get("value"),
                    target_chunk_id=event_dict.get("target_chunk_id"),
                    client_timestamp=datetime.fromisoformat(event_dict["client_timestamp"])
                        if event_dict.get("client_timestamp") else None,
                )

                if result.accepted:
                    accepted += 1
                else:
                    rejected += 1
                    errors.append({
                        "response_id": event_dict["response_id"],
                        "error": result.error,
                    })

            except Exception as e:
                rejected += 1
                errors.append({
                    "response_id": event_dict.get("response_id", "unknown"),
                    "error": str(e),
                })

        return {
            "accepted": accepted,
            "rejected": rejected,
            "errors": errors,
        }

    def get_buffered_events(self) -> List[FeedbackEvent]:
        """Get buffered feedback events.

        Returns:
            List of buffered events
        """
        return list(self._buffer)

    def clear_buffer(self):
        """Clear the feedback buffer."""
        self._buffer.clear()

    async def flush_buffer(self) -> int:
        """Flush buffered events to Kafka.

        Returns:
            Number of events flushed
        """
        if not self._buffer:
            return 0

        await self._ensure_initialized()

        if not self._producer or not self._initialized:
            logger.warning("Cannot flush buffer: Kafka not available")
            return 0

        count = 0
        for event in self._buffer:
            try:
                await self._producer.send_and_wait(
                    self.config.feedback_topic,
                    event.to_dict(),
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to flush event: {e}")

        self._buffer = self._buffer[count:]
        logger.info(f"Flushed {count} events to Kafka")
        return count

    async def close(self):
        """Close the feedback service."""
        if self._producer:
            await self._producer.stop()
            self._initialized = False


# Singleton instance
_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    """Get the global feedback service instance."""
    global _service
    if _service is None:
        _service = FeedbackService()
    return _service
