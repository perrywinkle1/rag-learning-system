"""Checkpoint management with S3-compatible storage.

This module provides model checkpoint storage, versioning,
and retrieval with support for S3 and local storage.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a model checkpoint."""
    checkpoint_id: str
    model_version: str
    
    # Storage location
    storage_path: str
    storage_type: str  # local, s3
    
    # Model details
    model_name: str
    model_size_bytes: int = 0
    
    # Training info
    training_job_id: str = ""
    epoch: int = 0
    step: int = 0
    
    # Metrics at checkpoint
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Integrity
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "model_version": self.model_version,
            "storage_path": self.storage_path,
            "storage_type": self.storage_type,
            "model_name": self.model_name,
            "model_size_bytes": self.model_size_bytes,
            "training_job_id": self.training_job_id,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "config": self.config,
            "tags": self.tags,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class CheckpointManager:
    """Manages model checkpoints with S3 and local storage support.
    
    Features:
    - Save and load model checkpoints
    - Version management with metadata
    - S3-compatible storage backend
    - Automatic cleanup of old checkpoints
    - Integrity verification with checksums
    """
    
    def __init__(
        self,
        storage_type: str = "local",
        local_path: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "checkpoints",
        max_checkpoints: int = 10,
        s3_client: Optional[Any] = None,
    ):
        """Initialize checkpoint manager.
        
        Args:
            storage_type: Storage backend (local, s3)
            local_path: Path for local storage
            s3_bucket: S3 bucket name
            s3_prefix: S3 key prefix
            max_checkpoints: Maximum checkpoints to retain
            s3_client: Pre-configured S3 client (optional)
        """
        self.storage_type = storage_type
        self.local_path = Path(local_path or "./checkpoints")
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.max_checkpoints = max_checkpoints
        self.s3_client = s3_client
        
        # Initialize local storage
        if storage_type == "local":
            self.local_path.mkdir(parents=True, exist_ok=True)
            self.metadata_path = self.local_path / "metadata"
            self.metadata_path.mkdir(exist_ok=True)
        
        # Checkpoint registry
        self.checkpoints: Dict[str, CheckpointInfo] = {}
        self._load_registry()
        
        logger.info(
            f"CheckpointManager initialized: type={storage_type}, "
            f"max_checkpoints={max_checkpoints}"
        )
    
    def save(
        self,
        model: Any,
        model_version: str,
        model_name: str = "embedding_model",
        training_job_id: str = "",
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> CheckpointInfo:
        """Save a model checkpoint.
        
        Args:
            model: Model to save (SentenceTransformer or PyTorch model)
            model_version: Version string for the model
            model_name: Name of the model
            training_job_id: Associated training job ID
            epoch: Training epoch
            step: Training step
            metrics: Metrics at checkpoint time
            config: Training configuration
            tags: Tags for filtering
            
        Returns:
            CheckpointInfo for the saved checkpoint
        """
        import uuid
        
        checkpoint_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint directory name
        checkpoint_name = f"{model_name}_{model_version}_{timestamp}"
        
        # Save to temporary directory first
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / checkpoint_name
            tmp_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self._save_model(model, tmp_path)
            
            # Compute checksum
            checksum = self._compute_checksum(tmp_path)
            
            # Compute size
            model_size = self._get_dir_size(tmp_path)
            
            # Determine storage path
            if self.storage_type == "local":
                storage_path = str(self.local_path / checkpoint_name)
                # Move to final location
                if Path(storage_path).exists():
                    shutil.rmtree(storage_path)
                shutil.copytree(tmp_path, storage_path)
            else:
                storage_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/{checkpoint_name}"
                self._upload_to_s3(tmp_path, checkpoint_name)
        
        # Create checkpoint info
        info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            model_version=model_version,
            storage_path=storage_path,
            storage_type=self.storage_type,
            model_name=model_name,
            model_size_bytes=model_size,
            training_job_id=training_job_id,
            epoch=epoch,
            step=step,
            metrics=metrics or {},
            config=config or {},
            tags=tags or [],
            checksum=checksum,
        )
        
        # Register checkpoint
        self.checkpoints[checkpoint_id] = info
        self._save_registry()
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(
            f"Saved checkpoint {checkpoint_id}: {storage_path} "
            f"({model_size / 1024 / 1024:.1f} MB)"
        )
        
        return info
    
    def load(
        self,
        checkpoint_id: str,
        device: str = "cpu",
        verify_checksum: bool = True,
    ) -> Any:
        """Load a model from checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to load
            device: Device to load model to
            verify_checksum: Whether to verify integrity
            
        Returns:
            Loaded model
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        info = self.checkpoints[checkpoint_id]
        
        # Get checkpoint to local path
        if self.storage_type == "local":
            local_path = Path(info.storage_path)
        else:
            local_path = self._download_from_s3(info)
        
        # Verify checksum
        if verify_checksum:
            current_checksum = self._compute_checksum(local_path)
            if current_checksum != info.checksum:
                raise ValueError(
                    f"Checkpoint corrupted: expected {info.checksum}, "
                    f"got {current_checksum}"
                )
        
        # Load model
        model = self._load_model(local_path, device)
        
        logger.info(f"Loaded checkpoint {checkpoint_id} from {info.storage_path}")
        
        return model
    
    def get_latest(self, model_name: Optional[str] = None) -> Optional[CheckpointInfo]:
        """Get the latest checkpoint.
        
        Args:
            model_name: Filter by model name
            
        Returns:
            Latest CheckpointInfo or None
        """
        candidates = list(self.checkpoints.values())
        
        if model_name:
            candidates = [c for c in candidates if c.model_name == model_name]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda c: c.created_at)
    
    def get_best(
        self,
        metric: str = "mrr",
        model_name: Optional[str] = None,
        higher_is_better: bool = True,
    ) -> Optional[CheckpointInfo]:
        """Get the best checkpoint by metric.
        
        Args:
            metric: Metric name to compare
            model_name: Filter by model name
            higher_is_better: Whether higher metric is better
            
        Returns:
            Best CheckpointInfo or None
        """
        candidates = [
            c for c in self.checkpoints.values()
            if metric in c.metrics
        ]
        
        if model_name:
            candidates = [c for c in candidates if c.model_name == model_name]
        
        if not candidates:
            return None
        
        if higher_is_better:
            return max(candidates, key=lambda c: c.metrics[metric])
        else:
            return min(candidates, key=lambda c: c.metrics[metric])
    
    def list_checkpoints(
        self,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[CheckpointInfo]:
        """List checkpoints with optional filtering.
        
        Args:
            model_name: Filter by model name
            tags: Filter by tags (any match)
            limit: Maximum number to return
            
        Returns:
            List of CheckpointInfo
        """
        candidates = list(self.checkpoints.values())
        
        if model_name:
            candidates = [c for c in candidates if c.model_name == model_name]
        
        if tags:
            candidates = [
                c for c in candidates
                if any(t in c.tags for t in tags)
            ]
        
        # Sort by creation time (newest first)
        candidates.sort(key=lambda c: c.created_at, reverse=True)
        
        return candidates[:limit]
    
    def delete(self, checkpoint_id: str):
        """Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        info = self.checkpoints[checkpoint_id]
        
        # Delete from storage
        if self.storage_type == "local":
            path = Path(info.storage_path)
            if path.exists():
                shutil.rmtree(path)
        else:
            self._delete_from_s3(info)
        
        # Remove from registry
        del self.checkpoints[checkpoint_id]
        self._save_registry()
        
        logger.info(f"Deleted checkpoint {checkpoint_id}")
    
    def _save_model(self, model: Any, path: Path):
        """Save model to path."""
        import torch
        
        try:
            from sentence_transformers import SentenceTransformer
            if isinstance(model, SentenceTransformer):
                model.save(str(path))
                return
        except ImportError:
            pass
        
        # Fallback to PyTorch save
        torch.save(model.state_dict(), path / "model.pt")
    
    def _load_model(self, path: Path, device: str) -> Any:
        """Load model from path."""
        import torch
        
        try:
            from sentence_transformers import SentenceTransformer
            if (path / "modules.json").exists():
                model = SentenceTransformer(str(path), device=device)
                return model
        except ImportError:
            pass
        
        # Fallback to PyTorch load
        state_dict = torch.load(path / "model.pt", map_location=device)
        return state_dict
    
    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of directory."""
        hasher = hashlib.sha256()
        
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()[:16]
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total
    
    def _upload_to_s3(self, local_path: Path, name: str):
        """Upload checkpoint to S3."""
        if not self.s3_client:
            self._init_s3_client()
        
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                key = f"{self.s3_prefix}/{name}/{file_path.relative_to(local_path)}"
                self.s3_client.upload_file(str(file_path), self.s3_bucket, key)
    
    def _download_from_s3(self, info: CheckpointInfo) -> Path:
        """Download checkpoint from S3 to local cache."""
        if not self.s3_client:
            self._init_s3_client()
        
        # Parse S3 path
        parts = info.storage_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1]
        
        # Create local cache directory
        cache_path = self.local_path / ".cache" / info.checkpoint_id
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # List and download objects
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            local_file = cache_path / Path(key).relative_to(prefix)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(bucket, key, str(local_file))
        
        return cache_path
    
    def _delete_from_s3(self, info: CheckpointInfo):
        """Delete checkpoint from S3."""
        if not self.s3_client:
            self._init_s3_client()
        
        parts = info.storage_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1]
        
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in response.get("Contents", []):
            self.s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
    
    def _init_s3_client(self):
        """Initialize S3 client."""
        try:
            import boto3
            self.s3_client = boto3.client("s3")
        except ImportError:
            raise ImportError("boto3 required for S3 storage")
    
    def _load_registry(self):
        """Load checkpoint registry from disk."""
        if self.storage_type != "local":
            return
        
        registry_file = self.metadata_path / "registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                data = json.load(f)
                for item in data:
                    info = CheckpointInfo.from_dict(item)
                    self.checkpoints[info.checkpoint_id] = info
    
    def _save_registry(self):
        """Save checkpoint registry to disk."""
        if self.storage_type != "local":
            return
        
        registry_file = self.metadata_path / "registry.json"
        data = [info.to_dict() for info in self.checkpoints.values()]
        
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by creation time
        sorted_checkpoints = sorted(
            self.checkpoints.values(),
            key=lambda c: c.created_at,
        )
        
        # Delete oldest until within limit
        to_delete = sorted_checkpoints[:len(sorted_checkpoints) - self.max_checkpoints]
        
        for info in to_delete:
            try:
                self.delete(info.checkpoint_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint {info.checkpoint_id}: {e}")

