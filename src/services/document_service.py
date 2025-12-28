"""Document management service.

This module provides functionality for ingesting, managing, and
deleting documents in the knowledge base.
"""

import logging
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from src.config import get_settings
from src.core.retrieval import VectorRetriever, get_retriever

logger = logging.getLogger(__name__)


class DocumentStatus(str, Enum):
    """Status of a document."""
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class ChunkingStrategy(str, Enum):
    """Chunking strategies for documents."""
    SEMANTIC = "semantic"
    FIXED = "fixed"
    SENTENCE = "sentence"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = 512  # Target tokens per chunk
    chunk_overlap: int = 50  # Overlap tokens


@dataclass
class Document:
    """A document in the knowledge base."""
    document_id: str
    external_id: Optional[str]
    title: str
    content: str
    source_type: str
    source_url: Optional[str]
    language: str
    metadata: Dict[str, Any]
    content_hash: str
    status: DocumentStatus
    chunks_count: int
    created_at: datetime
    updated_at: datetime
    embedding_model: Optional[str] = None


@dataclass
class DocumentChunk:
    """A chunk of a document."""
    chunk_id: str
    document_id: str
    content: str
    position: int
    metadata: Dict[str, Any]


@dataclass
class IngestResult:
    """Result of document ingestion."""
    document_id: str
    external_id: Optional[str]
    chunks_created: int
    status: DocumentStatus
    estimated_completion: Optional[datetime] = None
    job_id: Optional[str] = None
    poll_url: Optional[str] = None


class DocumentService:
    """Service for document management.

    Handles document ingestion, chunking, indexing, and deletion.
    """

    # Size thresholds
    ASYNC_THRESHOLD_BYTES = 100 * 1024  # 100KB
    MAX_DOCUMENT_BYTES = 10 * 1024 * 1024  # 10MB

    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
    ):
        """Initialize the document service.

        Args:
            retriever: Vector retrieval service for indexing
        """
        self.retriever = retriever or get_retriever()
        self.settings = get_settings()

        # In-memory document store (would use database in production)
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, List[DocumentChunk]] = {}

    def _hash_content(self, content: str) -> str:
        """Generate a hash of document content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _chunk_document(
        self,
        document_id: str,
        content: str,
        config: ChunkingConfig,
    ) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces.

        Args:
            document_id: Document identifier
            content: Document text
            config: Chunking configuration

        Returns:
            List of document chunks
        """
        if config.strategy == ChunkingStrategy.FIXED:
            return self._chunk_fixed(document_id, content, config)
        elif config.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentence(document_id, content, config)
        else:
            return self._chunk_semantic(document_id, content, config)

    def _chunk_fixed(
        self,
        document_id: str,
        content: str,
        config: ChunkingConfig,
    ) -> List[DocumentChunk]:
        """Chunk using fixed token count."""
        words = content.split()
        chunks = []
        position = 0

        i = 0
        while i < len(words):
            chunk_words = words[i:i + config.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append(DocumentChunk(
                chunk_id=f"{document_id}_{position}",
                document_id=document_id,
                content=chunk_text,
                position=position,
                metadata={"strategy": "fixed"},
            ))

            position += 1
            i += config.chunk_size - config.chunk_overlap

        return chunks

    def _chunk_sentence(
        self,
        document_id: str,
        content: str,
        config: ChunkingConfig,
    ) -> List[DocumentChunk]:
        """Chunk by sentences."""
        import re

        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = []
        current_size = 0
        position = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            if current_size + sentence_size > config.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(
                    chunk_id=f"{document_id}_{position}",
                    document_id=document_id,
                    content=chunk_text,
                    position=position,
                    metadata={"strategy": "sentence"},
                ))
                position += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add remaining content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id=f"{document_id}_{position}",
                document_id=document_id,
                content=chunk_text,
                position=position,
                metadata={"strategy": "sentence"},
            ))

        return chunks

    def _chunk_semantic(
        self,
        document_id: str,
        content: str,
        config: ChunkingConfig,
    ) -> List[DocumentChunk]:
        """Chunk semantically (using paragraphs as a proxy)."""
        import re

        # Split by paragraphs
        paragraphs = re.split(r'\n\n+', content)
        chunks = []
        current_chunk = []
        current_size = 0
        position = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_size = len(paragraph.split())

            # If a single paragraph is too large, split it
            if paragraph_size > config.chunk_size:
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{document_id}_{position}",
                        document_id=document_id,
                        content=chunk_text,
                        position=position,
                        metadata={"strategy": "semantic"},
                    ))
                    position += 1
                    current_chunk = []
                    current_size = 0

                # Use sentence chunking for large paragraphs
                sub_chunks = self._chunk_sentence(
                    f"{document_id}_{position}",
                    paragraph,
                    config,
                )
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = f"{document_id}_{position}"
                    sub_chunk.position = position
                    chunks.append(sub_chunk)
                    position += 1
                continue

            if current_size + paragraph_size > config.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(DocumentChunk(
                    chunk_id=f"{document_id}_{position}",
                    document_id=document_id,
                    content=chunk_text,
                    position=position,
                    metadata={"strategy": "semantic"},
                ))
                position += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(paragraph)
            current_size += paragraph_size

        # Add remaining content
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id=f"{document_id}_{position}",
                document_id=document_id,
                content=chunk_text,
                position=position,
                metadata={"strategy": "semantic"},
            ))

        return chunks

    async def ingest(
        self,
        title: str,
        content: str,
        source_type: str,
        external_id: Optional[str] = None,
        source_url: Optional[str] = None,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None,
        chunking: Optional[ChunkingConfig] = None,
    ) -> IngestResult:
        """Ingest a new document.

        Args:
            title: Document title
            content: Document text content
            source_type: Source type (upload, crawl, api, manual)
            external_id: External system ID
            source_url: Source URL
            language: Language code (ISO 639-1)
            metadata: Additional metadata
            chunking: Chunking configuration

        Returns:
            IngestResult with document info
        """
        # Validate size
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > self.MAX_DOCUMENT_BYTES:
            raise ValueError(f"Document exceeds {self.MAX_DOCUMENT_BYTES} byte limit")

        # Generate IDs
        document_id = str(uuid.uuid4())
        content_hash = self._hash_content(content)

        # Check for duplicates
        for doc in self._documents.values():
            if doc.content_hash == content_hash:
                raise ValueError(f"Duplicate document: {doc.document_id}")

        # Create document record
        now = datetime.utcnow()
        document = Document(
            document_id=document_id,
            external_id=external_id,
            title=title,
            content=content,
            source_type=source_type,
            source_url=source_url,
            language=language,
            metadata=metadata or {},
            content_hash=content_hash,
            status=DocumentStatus.PROCESSING,
            chunks_count=0,
            created_at=now,
            updated_at=now,
        )

        self._documents[document_id] = document

        # Chunk the document
        chunking_config = chunking or ChunkingConfig()
        chunks = self._chunk_document(document_id, content, chunking_config)
        self._chunks[document_id] = chunks

        # Index chunks
        chunk_dicts = [
            {
                "id": chunk.chunk_id,
                "content": chunk.content,
                "document_id": document_id,
                "metadata": {
                    **chunk.metadata,
                    "title": title,
                    "source_type": source_type,
                },
            }
            for chunk in chunks
        ]

        try:
            await self.retriever.index_chunks(chunk_dicts)
            document.status = DocumentStatus.INDEXED
            document.chunks_count = len(chunks)
            document.embedding_model = self.settings.embedding.model_name

            logger.info(f"Document indexed: {document_id} ({len(chunks)} chunks)")

        except Exception as e:
            document.status = DocumentStatus.FAILED
            logger.error(f"Failed to index document {document_id}: {e}")
            raise

        return IngestResult(
            document_id=document_id,
            external_id=external_id,
            chunks_created=len(chunks),
            status=document.status,
        )

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document or None
        """
        return self._documents.get(document_id)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document.

        Args:
            document_id: Document identifier

        Returns:
            True if deleted
        """
        if document_id not in self._documents:
            return False

        # Delete from vector store
        await self.retriever.delete_document(document_id)

        # Delete from local store
        del self._documents[document_id]
        if document_id in self._chunks:
            del self._chunks[document_id]

        logger.info(f"Document deleted: {document_id}")
        return True

    async def list_documents(
        self,
        page: int = 1,
        per_page: int = 20,
        source_type: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List documents with filtering.

        Args:
            page: Page number
            per_page: Items per page
            source_type: Filter by source type
            created_after: Filter by creation date
            created_before: Filter by creation date
            search: Search in titles

        Returns:
            Paginated document list
        """
        # Filter documents
        filtered = []
        for doc in self._documents.values():
            if source_type and doc.source_type != source_type:
                continue
            if created_after and doc.created_at < created_after:
                continue
            if created_before and doc.created_at > created_before:
                continue
            if search and search.lower() not in doc.title.lower():
                continue
            filtered.append(doc)

        # Sort by created_at desc
        filtered.sort(key=lambda d: d.created_at, reverse=True)

        # Paginate
        total = len(filtered)
        start = (page - 1) * per_page
        end = start + per_page
        paginated = filtered[start:end]

        return {
            "documents": [
                {
                    "document_id": d.document_id,
                    "external_id": d.external_id,
                    "title": d.title,
                    "source_type": d.source_type,
                    "status": d.status.value,
                    "chunks_count": d.chunks_count,
                    "created_at": d.created_at.isoformat(),
                    "updated_at": d.updated_at.isoformat(),
                }
                for d in paginated
            ],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page,
            },
        }


# Singleton instance
_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    global _service
    if _service is None:
        _service = DocumentService()
    return _service
