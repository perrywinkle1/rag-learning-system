"""Query preprocessing and expansion.

This module provides functionality to preprocess queries before
retrieval, including normalization, expansion, and intent detection.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """A processed query with expansion and metadata."""
    original: str
    normalized: str
    tokens: List[str]
    expanded_terms: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expanded_query(self) -> str:
        """Get query with expanded terms."""
        if self.expanded_terms:
            return f"{self.normalized} {' '.join(self.expanded_terms)}"
        return self.normalized


class QueryProcessor:
    """Service for preprocessing and expanding queries.

    Performs normalization, stopword removal, synonym expansion,
    and basic intent detection.
    """

    # Common English stopwords
    STOPWORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "but", "and", "or", "if", "because",
        "until", "while", "about", "against", "this", "that",
    }

    # Simple synonym dictionary (could be loaded from file or database)
    SYNONYMS: Dict[str, List[str]] = {
        "buy": ["purchase", "acquire", "get"],
        "sell": ["offer", "provide"],
        "help": ["assist", "support", "aid"],
        "problem": ["issue", "error", "bug"],
        "fix": ["solve", "resolve", "repair"],
        "fast": ["quick", "rapid", "speedy"],
        "slow": ["sluggish", "delayed"],
        "error": ["exception", "failure", "bug"],
        "config": ["configuration", "settings", "options"],
        "auth": ["authentication", "authorization", "login"],
        "api": ["interface", "endpoint"],
    }

    # Intent patterns
    INTENT_PATTERNS = {
        "question": [
            r"^(what|who|where|when|why|how|which|whose)\s",
            r"\?$",
            r"^(can|could|would|is|are|do|does|did)\s.*\?",
        ],
        "action": [
            r"^(create|make|build|add|remove|delete|update|change|modify)\s",
            r"^(show|display|list|get|find|search)\s",
        ],
        "troubleshooting": [
            r"(error|exception|fail|problem|issue|bug|broken|not working)",
            r"(why.*not|cannot|can't|couldn't)",
        ],
        "howto": [
            r"^how (to|do|can|should)\s",
            r"(steps to|guide for|tutorial)",
        ],
    }

    def __init__(
        self,
        enable_expansion: bool = True,
        enable_stopword_removal: bool = False,
        max_expanded_terms: int = 3,
    ):
        """Initialize the query processor.

        Args:
            enable_expansion: Enable synonym expansion
            enable_stopword_removal: Remove stopwords from query
            max_expanded_terms: Maximum number of expanded terms to add
        """
        self.enable_expansion = enable_expansion
        self.enable_stopword_removal = enable_stopword_removal
        self.max_expanded_terms = max_expanded_terms

        # Compile intent patterns
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }

    def process(self, query: str) -> ProcessedQuery:
        """Process a query.

        Args:
            query: Raw query string

        Returns:
            ProcessedQuery with normalization and expansion
        """
        # Normalize
        normalized = self._normalize(query)

        # Tokenize
        tokens = self._tokenize(normalized)

        # Remove stopwords if enabled
        if self.enable_stopword_removal:
            tokens = [t for t in tokens if t.lower() not in self.STOPWORDS]

        # Detect intent
        intent = self._detect_intent(query)

        # Extract entities (placeholder - would use NER)
        entities = self._extract_entities(query)

        # Expand terms
        expanded_terms = []
        if self.enable_expansion:
            expanded_terms = self._expand_terms(tokens)

        return ProcessedQuery(
            original=query,
            normalized=normalized,
            tokens=tokens,
            expanded_terms=expanded_terms,
            intent=intent,
            entities=entities,
        )

    def _normalize(self, query: str) -> str:
        """Normalize query text.

        - Lowercase
        - Remove extra whitespace
        - Basic punctuation handling
        """
        # Strip whitespace
        normalized = query.strip()

        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Keep alphanumeric, spaces, and basic punctuation
        # normalized = re.sub(r"[^\w\s\-\.\?\!]", "", normalized)

        return normalized

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _detect_intent(self, query: str) -> Optional[str]:
        """Detect query intent.

        Returns:
            Intent category or None
        """
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent
        return None

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from query.

        This is a placeholder - would use a proper NER model.
        """
        entities: Dict[str, List[str]] = {}

        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            entities["quoted"] = quoted

        # Extract code-like tokens (e.g., function names, file paths)
        code_tokens = re.findall(r"[\w_]+\(|\.\w+|\w+\.\w+", query)
        if code_tokens:
            entities["code"] = code_tokens

        return entities

    def _expand_terms(self, tokens: List[str]) -> List[str]:
        """Expand tokens with synonyms.

        Args:
            tokens: Query tokens

        Returns:
            List of expansion terms
        """
        expanded = []

        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.SYNONYMS:
                # Add synonyms (limit to max)
                synonyms = self.SYNONYMS[token_lower]
                remaining = self.max_expanded_terms - len(expanded)
                expanded.extend(synonyms[:remaining])

                if len(expanded) >= self.max_expanded_terms:
                    break

        return expanded

    def add_synonym(self, word: str, synonyms: List[str]):
        """Add a synonym mapping.

        Args:
            word: Base word
            synonyms: List of synonyms
        """
        self.SYNONYMS[word.lower()] = synonyms


class QueryRewriter:
    """Rewrites queries for better retrieval.

    Uses rules and optionally LLM to reformulate queries.
    """

    def __init__(self, llm_client=None):
        """Initialize the rewriter.

        Args:
            llm_client: Optional LLM client for query rewriting
        """
        self.llm_client = llm_client

    async def rewrite(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """Rewrite a query for better retrieval.

        Args:
            query: Original query
            context: Optional context for rewriting

        Returns:
            Rewritten query
        """
        if self.llm_client is None:
            # Basic rule-based rewriting
            return self._rule_based_rewrite(query)

        # LLM-based rewriting
        return await self._llm_rewrite(query, context)

    def _rule_based_rewrite(self, query: str) -> str:
        """Apply rule-based query rewriting."""
        rewritten = query

        # Remove filler words at start
        filler_patterns = [
            r"^(please|kindly|i want to|i need to|can you|could you)\s+",
            r"^(tell me|show me|explain|describe)\s+",
        ]

        for pattern in filler_patterns:
            rewritten = re.sub(pattern, "", rewritten, flags=re.IGNORECASE)

        return rewritten.strip() or query

    async def _llm_rewrite(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """Use LLM to rewrite query."""
        # Would implement LLM-based rewriting here
        return query


# Singleton instance
_processor: Optional[QueryProcessor] = None


def get_query_processor() -> QueryProcessor:
    """Get the global query processor instance."""
    global _processor
    if _processor is None:
        _processor = QueryProcessor()
    return _processor
