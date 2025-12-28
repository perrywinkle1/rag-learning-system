"""PyTorch Dataset for contrastive learning.

This module provides dataset classes for training embedding models
with contrastive learning objectives.
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from src.learning.pair_generator import TrainingPair, TripletSample

logger = logging.getLogger(__name__)


class TrainingDataset(Dataset):
    """PyTorch Dataset for contrastive learning training.
    
    Supports multiple data formats:
    - TrainingPair objects for pair-based learning
    - TripletSample objects for triplet loss
    - Raw text tuples (anchor, positive, negative)
    """
    
    def __init__(
        self,
        data: Union[List[TrainingPair], List[TripletSample], List[Tuple[str, str, str]]],
        tokenizer: Optional[Any] = None,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
        use_embeddings: bool = False,
        augment: bool = False,
        augmentation_fn: Optional[Callable] = None,
    ):
        """Initialize dataset.
        
        Args:
            data: Training data (pairs, triplets, or text tuples)
            tokenizer: Pre-initialized tokenizer
            tokenizer_name: Name of tokenizer to load if not provided
            max_length: Maximum sequence length
            use_embeddings: If True, use pre-computed embeddings
            augment: Whether to apply data augmentation
            augmentation_fn: Custom augmentation function
        """
        self.data = data
        self.max_length = max_length
        self.use_embeddings = use_embeddings
        self.augment = augment
        self.augmentation_fn = augmentation_fn
        
        # Initialize tokenizer
        if tokenizer:
            self.tokenizer = tokenizer
        elif AutoTokenizer and not use_embeddings:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        
        # Detect data type
        if data:
            if isinstance(data[0], TrainingPair):
                self.data_type = "pair"
            elif isinstance(data[0], TripletSample):
                self.data_type = "triplet"
            elif isinstance(data[0], tuple):
                self.data_type = "tuple"
            else:
                raise ValueError(f"Unsupported data type: {type(data[0])}")
        else:
            self.data_type = "unknown"
        
        logger.info(
            f"TrainingDataset initialized: {len(data)} samples, "
            f"type={self.data_type}, use_embeddings={use_embeddings}"
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.
        
        Returns dictionary with tokenized anchor, positive, and negative texts,
        or pre-computed embeddings if use_embeddings=True.
        """
        item = self.data[idx]
        
        if self.data_type == "pair":
            return self._process_pair(item)
        elif self.data_type == "triplet":
            return self._process_triplet(item)
        else:
            return self._process_tuple(item)
    
    def _process_pair(self, pair: TrainingPair) -> Dict[str, torch.Tensor]:
        """Process a TrainingPair."""
        if self.use_embeddings:
            return {
                "anchor_embedding": torch.tensor(pair.query_embedding, dtype=torch.float32),
                "positive_embedding": torch.tensor(pair.positive_chunk_embedding, dtype=torch.float32),
                "negative_embedding": torch.tensor(pair.negative_chunk_embedding, dtype=torch.float32),
                "positive_score": torch.tensor(pair.positive_score, dtype=torch.float32),
                "negative_score": torch.tensor(pair.negative_score, dtype=torch.float32),
            }
        
        # Get texts
        anchor_text = pair.query_text
        positive_text = pair.positive_chunk_text
        negative_text = pair.negative_chunk_text
        
        # Apply augmentation
        if self.augment and self.augmentation_fn:
            anchor_text = self.augmentation_fn(anchor_text)
        
        # Tokenize
        return self._tokenize_triplet(anchor_text, positive_text, negative_text)
    
    def _process_triplet(self, triplet: TripletSample) -> Dict[str, torch.Tensor]:
        """Process a TripletSample."""
        if self.use_embeddings:
            return {
                "anchor_embedding": torch.tensor(triplet.anchor_embedding, dtype=torch.float32),
                "positive_embedding": torch.tensor(triplet.positive_embedding, dtype=torch.float32),
                "negative_embedding": torch.tensor(triplet.negative_embedding, dtype=torch.float32),
                "margin": torch.tensor(triplet.margin, dtype=torch.float32),
            }
        
        anchor_text = triplet.anchor_text
        positive_text = triplet.positive_text
        negative_text = triplet.negative_text
        
        if self.augment and self.augmentation_fn:
            anchor_text = self.augmentation_fn(anchor_text)
        
        return self._tokenize_triplet(anchor_text, positive_text, negative_text)
    
    def _process_tuple(self, item: Tuple[str, str, str]) -> Dict[str, torch.Tensor]:
        """Process a text tuple."""
        anchor_text, positive_text, negative_text = item
        
        if self.augment and self.augmentation_fn:
            anchor_text = self.augmentation_fn(anchor_text)
        
        return self._tokenize_triplet(anchor_text, positive_text, negative_text)
    
    def _tokenize_triplet(
        self,
        anchor: str,
        positive: str,
        negative: str,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize anchor, positive, and negative texts."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text processing")
        
        # Tokenize each text
        anchor_enc = self.tokenizer(
            anchor,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        positive_enc = self.tokenizer(
            positive,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        negative_enc = self.tokenizer(
            negative,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "anchor_input_ids": anchor_enc["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_enc["attention_mask"].squeeze(0),
            "positive_input_ids": positive_enc["input_ids"].squeeze(0),
            "positive_attention_mask": positive_enc["attention_mask"].squeeze(0),
            "negative_input_ids": negative_enc["input_ids"].squeeze(0),
            "negative_attention_mask": negative_enc["attention_mask"].squeeze(0),
        }
    
    @classmethod
    def from_pairs(
        cls,
        pairs: List[TrainingPair],
        **kwargs,
    ) -> "TrainingDataset":
        """Create dataset from TrainingPair list."""
        return cls(data=pairs, **kwargs)
    
    @classmethod
    def from_triplets(
        cls,
        triplets: List[TripletSample],
        **kwargs,
    ) -> "TrainingDataset":
        """Create dataset from TripletSample list."""
        return cls(data=triplets, **kwargs)
    
    @classmethod
    def from_texts(
        cls,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        **kwargs,
    ) -> "TrainingDataset":
        """Create dataset from text lists."""
        if len(anchors) != len(positives) or len(anchors) != len(negatives):
            raise ValueError("All text lists must have same length")
        
        data = list(zip(anchors, positives, negatives))
        return cls(data=data, **kwargs)
    
    def split(
        self,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple["TrainingDataset", "TrainingDataset"]:
        """Split dataset into train and validation sets.
        
        Args:
            val_ratio: Fraction for validation set
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        random.seed(seed)
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        
        val_size = int(len(indices) * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_data = [self.data[i] for i in train_indices]
        val_data = [self.data[i] for i in val_indices]
        
        train_dataset = TrainingDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_embeddings=self.use_embeddings,
            augment=self.augment,
            augmentation_fn=self.augmentation_fn,
        )
        
        val_dataset = TrainingDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_embeddings=self.use_embeddings,
            augment=False,  # No augmentation for validation
            augmentation_fn=None,
        )
        
        logger.info(
            f"Split dataset: {len(train_data)} train, {len(val_data)} validation"
        )
        
        return train_dataset, val_dataset


class InBatchNegativesDataset(Dataset):
    """Dataset that uses in-batch negatives for efficient training.
    
    Each batch contains pairs of (query, positive_doc), and the
    positive documents of other queries in the batch serve as negatives.
    """
    
    def __init__(
        self,
        queries: List[str],
        positives: List[str],
        tokenizer: Optional[Any] = None,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
    ):
        """Initialize dataset.
        
        Args:
            queries: List of query texts
            positives: List of positive document texts
            tokenizer: Pre-initialized tokenizer
            tokenizer_name: Name of tokenizer to load
            max_length: Maximum sequence length
        """
        if len(queries) != len(positives):
            raise ValueError("queries and positives must have same length")
        
        self.queries = queries
        self.positives = positives
        self.max_length = max_length
        
        if tokenizer:
            self.tokenizer = tokenizer
        elif AutoTokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            raise ImportError("transformers required for tokenization")
        
        logger.info(f"InBatchNegativesDataset initialized: {len(queries)} pairs")
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single (query, positive) pair."""
        query = self.queries[idx]
        positive = self.positives[idx]
        
        query_enc = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        positive_enc = self.tokenizer(
            positive,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "positive_input_ids": positive_enc["input_ids"].squeeze(0),
            "positive_attention_mask": positive_enc["attention_mask"].squeeze(0),
        }


def default_augmentation(text: str) -> str:
    """Default text augmentation: random word dropout.
    
    Args:
        text: Input text
        
    Returns:
        Augmented text with randomly dropped words
    """
    if random.random() < 0.5:
        return text
    
    words = text.split()
    if len(words) <= 3:
        return text
    
    # Drop 10-20% of words
    drop_ratio = random.uniform(0.1, 0.2)
    num_drop = max(1, int(len(words) * drop_ratio))
    
    indices_to_drop = set(random.sample(range(len(words)), num_drop))
    augmented = [w for i, w in enumerate(words) if i not in indices_to_drop]
    
    return " ".join(augmented)

