"""Response generation using LLM.

This module provides functionality to generate answers using
retrieved documents as context.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncIterator
from abc import ABC, abstractmethod

from src.config import GenerationConfig, get_settings
from src.core.reranker import RankedCandidate

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation to a source document."""
    chunk_id: str
    document_id: str
    content_snippet: str
    rank: int
    score: float


@dataclass
class GeneratedResponse:
    """A generated response with citations."""
    response_id: str
    answer: str
    citations: List[Citation]
    confidence: float
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A chunk from streaming generation."""
    text: str
    is_final: bool = False
    citations: Optional[List[Citation]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate a response."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate using OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream generate using OpenAI."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate using Anthropic."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream generate using Anthropic."""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


class ResponseGenerator:
    """Service for generating responses using LLM.

    Combines retrieved documents with user query to generate
    grounded responses with citations.
    """

    PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}

User Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context documents.
2. If the context doesn't contain enough information to answer, say so.
3. Be concise and direct in your response.
4. When referencing information, cite the source using [1], [2], etc.

Answer:"""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize the generator.

        Args:
            config: Generation configuration
            llm_client: Pre-configured LLM client
        """
        self.config = config or get_settings().generation
        self._client = llm_client
        self._initialized = llm_client is not None

    def _ensure_initialized(self):
        """Lazy initialization of LLM client."""
        if not self._initialized:
            if self.config.provider == "openai":
                if not self.config.openai_api_key:
                    raise ValueError("OpenAI API key not configured")
                self._client = OpenAIClient(
                    api_key=self.config.openai_api_key,
                    model=self.config.model,
                )
            elif self.config.provider == "anthropic":
                if not self.config.anthropic_api_key:
                    raise ValueError("Anthropic API key not configured")
                self._client = AnthropicClient(
                    api_key=self.config.anthropic_api_key,
                    model=self.config.model,
                )
            else:
                raise ValueError(f"Unknown LLM provider: {self.config.provider}")

            self._initialized = True
            logger.info(f"Initialized {self.config.provider} client with model {self.config.model}")

    def _build_context(self, candidates: List[RankedCandidate]) -> str:
        """Build context string from candidates."""
        context_parts = []
        for i, candidate in enumerate(candidates, 1):
            context_parts.append(
                f"[{i}] (Document: {candidate.document_id}, Score: {candidate.final_score:.2f})\n"
                f"{candidate.content}\n"
            )
        return "\n".join(context_parts)

    def _build_prompt(self, query: str, candidates: List[RankedCandidate]) -> str:
        """Build the prompt for generation."""
        context = self._build_context(candidates)
        return self.PROMPT_TEMPLATE.format(context=context, query=query)

    def _extract_citations(
        self,
        answer: str,
        candidates: List[RankedCandidate],
    ) -> List[Citation]:
        """Extract citations from the answer."""
        import re

        # Find citation references like [1], [2], etc.
        citation_refs = set(re.findall(r"\[(\d+)\]", answer))

        citations = []
        for ref in citation_refs:
            idx = int(ref) - 1
            if 0 <= idx < len(candidates):
                candidate = candidates[idx]
                citations.append(Citation(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    content_snippet=candidate.content[:200],
                    rank=candidate.rank,
                    score=candidate.final_score,
                ))

        return citations

    async def generate(
        self,
        query: str,
        candidates: List[RankedCandidate],
        response_id: str,
    ) -> GeneratedResponse:
        """Generate a response.

        Args:
            query: User query
            candidates: Retrieved and reranked candidates
            response_id: Unique response identifier

        Returns:
            GeneratedResponse with answer and citations
        """
        import time
        import uuid

        start_time = time.time()

        self._ensure_initialized()

        if not candidates:
            return GeneratedResponse(
                response_id=response_id,
                answer="I couldn't find any relevant information to answer your question.",
                citations=[],
                confidence=0.0,
                model=self.config.model,
            )

        # Build prompt
        prompt = self._build_prompt(query, candidates)

        # Generate
        answer = await self._client.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Extract citations
        citations = self._extract_citations(answer, candidates)

        # Calculate confidence based on source scores
        avg_score = sum(c.final_score for c in candidates[:3]) / min(3, len(candidates))
        confidence = min(1.0, avg_score)

        latency_ms = (time.time() - start_time) * 1000

        return GeneratedResponse(
            response_id=response_id,
            answer=answer,
            citations=citations,
            confidence=confidence,
            model=self.config.model,
            latency_ms=latency_ms,
        )

    async def generate_stream(
        self,
        query: str,
        candidates: List[RankedCandidate],
        response_id: str,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response.

        Args:
            query: User query
            candidates: Retrieved candidates
            response_id: Response identifier

        Yields:
            StreamChunk objects
        """
        self._ensure_initialized()

        if not candidates:
            yield StreamChunk(
                text="I couldn't find any relevant information to answer your question.",
                is_final=True,
                citations=[],
            )
            return

        # Build prompt
        prompt = self._build_prompt(query, candidates)

        # Stream response
        full_answer = ""
        async for chunk in self._client.generate_stream(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ):
            full_answer += chunk
            yield StreamChunk(text=chunk, is_final=False)

        # Final chunk with citations
        citations = self._extract_citations(full_answer, candidates)
        yield StreamChunk(
            text="",
            is_final=True,
            citations=citations,
        )


# Singleton instance
_generator: Optional[ResponseGenerator] = None


def get_generator() -> ResponseGenerator:
    """Get the global generator instance."""
    global _generator
    if _generator is None:
        _generator = ResponseGenerator()
    return _generator
