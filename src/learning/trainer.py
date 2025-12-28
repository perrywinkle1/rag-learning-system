"""Embedding model trainer with contrastive learning.

This module provides the training infrastructure for fine-tuning
embedding models using contrastive learning objectives.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from src.learning.dataset import TrainingDataset
from src.learning.losses import (
    TripletMarginLoss,
    InfoNCELoss,
    MultipleNegativesRankingLoss,
    ContrastiveLoss,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Training objective
    objective: str = "contrastive"  # contrastive, triplet, infonce, mnrl
    
    # Model configuration
    base_model: str = "all-MiniLM-L6-v2"
    max_seq_length: int = 512
    pooling_strategy: str = "mean"  # mean, cls, max
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Loss configuration
    margin: float = 0.2  # For triplet/contrastive loss
    temperature: float = 0.05  # For InfoNCE
    
    # Validation
    validation_split: float = 0.1
    eval_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    output_dir: str = "./checkpoints"
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    logging_steps: int = 10
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_objectives = {"contrastive", "triplet", "infonce", "mnrl"}
        if self.objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")
        
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


@dataclass
class TrainingResult:
    """Result of a training run."""
    job_id: str
    model_version: str
    
    # Training metrics
    training_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    # Final metrics
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    
    # Training details
    total_steps: int = 0
    total_samples: int = 0
    epochs_completed: int = 0
    early_stopped: bool = False
    
    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Checkpoints
    checkpoint_path: str = ""
    best_checkpoint_path: str = ""
    
    # Configuration used
    config: Optional[TrainingConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "model_version": self.model_version,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_steps": self.total_steps,
            "total_samples": self.total_samples,
            "epochs_completed": self.epochs_completed,
            "early_stopped": self.early_stopped,
            "duration_seconds": self.duration_seconds,
            "checkpoint_path": self.checkpoint_path,
        }


class EmbeddingTrainer:
    """Trainer for embedding models with contrastive learning.
    
    Supports multiple training objectives:
    - Contrastive: Binary classification of positive/negative pairs
    - Triplet: Triplet margin loss with anchor, positive, negative
    - InfoNCE: Info noise contrastive estimation
    - MNRL: Multiple negatives ranking loss
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[Any] = None,
    ):
        """Initialize trainer.
        
        Args:
            config: Training configuration
            model: Pre-loaded model (optional)
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        self._set_seed(config.seed)
        
        # Load or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = self._load_base_model()
        
        # Initialize loss function
        self.loss_fn = self._create_loss_function()
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.current_epoch = 0
        
        logger.info(
            f"EmbeddingTrainer initialized with {config.objective} objective "
            f"on {config.device}"
        )
    
    def train(
        self,
        train_dataset: TrainingDataset,
        val_dataset: Optional[TrainingDataset] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> TrainingResult:
        """Train the embedding model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            callbacks: Optional callbacks for training events
            
        Returns:
            TrainingResult with metrics and checkpoints
        """
        job_id = str(uuid.uuid4())[:8]
        model_version = f"v_{int(time.time())}"
        
        result = TrainingResult(
            job_id=job_id,
            model_version=model_version,
            config=self.config,
        )
        
        logger.info(f"Starting training job {job_id}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        # Calculate training steps
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = self.config.warmup_steps or int(total_steps * self.config.warmup_ratio)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(total_steps, warmup_steps)
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Training epoch
                train_loss = self._train_epoch(train_loader, result)
                result.training_loss.append(train_loss)
                
                # Validation
                if val_loader:
                    val_loss = self._validate(val_loader)
                    result.validation_loss.append(val_loss)
                    
                    # Early stopping check
                    if val_loss < best_val_loss - self.config.early_stopping_threshold:
                        best_val_loss = val_loss
                        result.best_val_loss = val_loss
                        result.best_epoch = epoch
                        patience_counter = 0
                        
                        # Save best checkpoint
                        best_path = self._save_checkpoint(f"best_{job_id}")
                        result.best_checkpoint_path = best_path
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        result.early_stopped = True
                        break
                
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"train_loss={train_loss:.4f}"
                    + (f", val_loss={val_loss:.4f}" if val_loader else "")
                )
                
                result.epochs_completed = epoch + 1
            
            # Final checkpoint
            final_path = self._save_checkpoint(f"final_{job_id}")
            result.checkpoint_path = final_path
            
            # Final metrics
            result.final_train_loss = result.training_loss[-1] if result.training_loss else 0.0
            result.final_val_loss = result.validation_loss[-1] if result.validation_loss else 0.0
            result.total_steps = self.global_step
            result.total_samples = len(train_dataset) * result.epochs_completed
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        logger.info(
            f"Training completed: job_id={job_id}, "
            f"epochs={result.epochs_completed}, "
            f"best_val_loss={result.best_val_loss:.4f}"
        )
        
        return result
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        result: TrainingResult,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % self.config.logging_steps == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                result.learning_rates.append(lr)
                
                logger.debug(
                    f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                )
        
        return total_loss / num_batches
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single training step."""
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        loss = self._compute_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        if self.config.objective == "triplet":
            anchor_emb = self._encode(batch["anchor_input_ids"], batch.get("anchor_attention_mask"))
            positive_emb = self._encode(batch["positive_input_ids"], batch.get("positive_attention_mask"))
            negative_emb = self._encode(batch["negative_input_ids"], batch.get("negative_attention_mask"))
            return self.loss_fn(anchor_emb, positive_emb, negative_emb)
        
        elif self.config.objective in ("contrastive", "mnrl"):
            anchor_emb = self._encode(batch["anchor_input_ids"], batch.get("anchor_attention_mask"))
            positive_emb = self._encode(batch["positive_input_ids"], batch.get("positive_attention_mask"))
            
            if "negative_input_ids" in batch:
                negative_emb = self._encode(batch["negative_input_ids"], batch.get("negative_attention_mask"))
                return self.loss_fn(anchor_emb, positive_emb, negative_emb)
            else:
                return self.loss_fn(anchor_emb, positive_emb)
        
        elif self.config.objective == "infonce":
            anchor_emb = self._encode(batch["anchor_input_ids"], batch.get("anchor_attention_mask"))
            positive_emb = self._encode(batch["positive_input_ids"], batch.get("positive_attention_mask"))
            return self.loss_fn(anchor_emb, positive_emb)
        
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")
    
    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text using the model."""
        if SentenceTransformer and isinstance(self.model, SentenceTransformer):
            features = {"input_ids": input_ids}
            if attention_mask is not None:
                features["attention_mask"] = attention_mask
            return self.model(features)["sentence_embedding"]
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            if self.config.pooling_strategy == "cls":
                return outputs.last_hidden_state[:, 0, :]
            elif self.config.pooling_strategy == "mean":
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    return sum_embeddings / sum_mask
                return outputs.last_hidden_state.mean(dim=1)
            else:
                return outputs.last_hidden_state.max(dim=1)[0]
    
    def _load_base_model(self) -> Any:
        """Load base embedding model."""
        if SentenceTransformer:
            model = SentenceTransformer(self.config.base_model)
            model.max_seq_length = self.config.max_seq_length
            model.to(self.device)
            return model
        else:
            raise ImportError("sentence-transformers required")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on objective."""
        if self.config.objective == "triplet":
            return TripletMarginLoss(margin=self.config.margin)
        elif self.config.objective == "contrastive":
            return ContrastiveLoss(margin=self.config.margin)
        elif self.config.objective == "infonce":
            return InfoNCELoss(temperature=self.config.temperature)
        elif self.config.objective == "mnrl":
            return MultipleNegativesRankingLoss(scale=1.0 / self.config.temperature)
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def _create_scheduler(
        self,
        total_steps: int,
        warmup_steps: int,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup."""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )
    
    def _save_checkpoint(self, name: str) -> str:
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = output_dir / name
        
        if SentenceTransformer and isinstance(self.model, SentenceTransformer):
            self.model.save(str(checkpoint_path))
        else:
            torch.save(self.model.state_dict(), str(checkpoint_path / "model.pt"))
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

