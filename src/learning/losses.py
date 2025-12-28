"""Custom loss functions for contrastive learning.

This module provides loss functions for training embedding models
with various contrastive learning objectives.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TripletMarginLoss(nn.Module):
    """Triplet margin loss for embedding learning.
    
    Encourages the anchor to be closer to the positive than to the negative
    by at least a margin.
    
    Loss = max(0, margin + d(anchor, positive) - d(anchor, negative))
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        distance: str = "cosine",
        reduction: str = "mean",
    ):
        """Initialize triplet margin loss.
        
        Args:
            margin: Margin between positive and negative distances
            distance: Distance metric (cosine, euclidean)
            reduction: Reduction method (mean, sum, none)
        """
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction
        
        logger.debug(f"TripletMarginLoss initialized: margin={margin}, distance={distance}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet margin loss.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss tensor
        """
        # Normalize for cosine distance
        if self.distance == "cosine":
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            
            # Cosine distance = 1 - cosine_similarity
            pos_dist = 1 - (anchor * positive).sum(dim=1)
            neg_dist = 1 - (anchor * negative).sum(dim=1)
        else:
            # Euclidean distance
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
        
        # Triplet loss
        loss = F.relu(self.margin + pos_dist - neg_dist)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class InfoNCELoss(nn.Module):
    """InfoNCE (Noise Contrastive Estimation) loss.
    
    Also known as NT-Xent loss. Uses in-batch negatives for efficient
    contrastive learning.
    
    Loss = -log(exp(sim(anchor, positive)/temp) / sum(exp(sim(anchor, all)/temp)))
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean",
    ):
        """Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature scaling factor
            reduction: Reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
        logger.debug(f"InfoNCELoss initialized: temperature={temperature}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss with in-batch negatives.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss tensor
        """
        batch_size = anchor.shape[0]
        
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Compute similarity matrix
        # Each anchor is compared against all positives (including its own)
        similarity = torch.matmul(anchor, positive.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=anchor.device)
        
        # Cross entropy loss (treats it as classification)
        loss = F.cross_entropy(similarity, labels, reduction=self.reduction)
        
        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """Multiple Negatives Ranking Loss (MNRL).
    
    Similar to InfoNCE but specifically designed for sentence embeddings.
    Uses all other positives in the batch as negatives.
    
    This is the loss used by sentence-transformers for training.
    """
    
    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct: str = "cosine",
        reduction: str = "mean",
    ):
        """Initialize MNRL loss.
        
        Args:
            scale: Scale factor for similarities
            similarity_fct: Similarity function (cosine, dot)
            reduction: Reduction method
        """
        super().__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.reduction = reduction
        
        logger.debug(f"MultipleNegativesRankingLoss initialized: scale={scale}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute MNRL loss.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Optional hard negatives (batch_size, embedding_dim)
            
        Returns:
            Loss tensor
        """
        if self.similarity_fct == "cosine":
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            if negative is not None:
                negative = F.normalize(negative, p=2, dim=1)
        
        # Compute scores against all positives
        scores = torch.matmul(anchor, positive.T) * self.scale
        
        # Add hard negatives if provided
        if negative is not None:
            neg_scores = (anchor * negative).sum(dim=1, keepdim=True) * self.scale
            scores = torch.cat([scores, neg_scores], dim=1)
        
        # Labels: diagonal is the correct positive
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        
        loss = F.cross_entropy(scores, labels, reduction=self.reduction)
        
        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for binary classification of pairs.
    
    Pulls together positive pairs and pushes apart negative pairs.
    
    Loss = (1-y) * d^2 + y * max(0, margin - d)^2
    where y=0 for positive pairs, y=1 for negative pairs
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance: str = "cosine",
        reduction: str = "mean",
    ):
        """Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
            distance: Distance metric
            reduction: Reduction method
        """
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction
        
        logger.debug(f"ContrastiveLoss initialized: margin={margin}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Optional negative embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss tensor
        """
        if self.distance == "cosine":
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            
            # Cosine distance for positive pairs (should be small)
            pos_dist = 1 - (anchor * positive).sum(dim=1)
            pos_loss = pos_dist ** 2
        else:
            pos_dist = F.pairwise_distance(anchor, positive)
            pos_loss = pos_dist ** 2
        
        # Negative pairs if provided
        if negative is not None:
            if self.distance == "cosine":
                negative = F.normalize(negative, p=2, dim=1)
                neg_dist = 1 - (anchor * negative).sum(dim=1)
            else:
                neg_dist = F.pairwise_distance(anchor, negative)
            
            neg_loss = F.relu(self.margin - neg_dist) ** 2
            loss = pos_loss + neg_loss
        else:
            loss = pos_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CircleLoss(nn.Module):
    """Circle loss for flexible optimization.
    
    Provides self-paced learning by adjusting the weight of each similarity
    score based on its current optimization status.
    
    Reference: Circle Loss: A Unified Perspective of Pair Similarity Optimization
    """
    
    def __init__(
        self,
        margin: float = 0.25,
        gamma: float = 256,
        reduction: str = "mean",
    ):
        """Initialize circle loss.
        
        Args:
            margin: Relaxation factor
            gamma: Scale factor
            reduction: Reduction method
        """
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        
        self.O_p = 1 + margin
        self.O_n = -margin
        self.Delta_p = 1 - margin
        self.Delta_n = margin
        
        logger.debug(f"CircleLoss initialized: margin={margin}, gamma={gamma}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute circle loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            Loss tensor
        """
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute similarities
        sim_pos = (anchor * positive).sum(dim=1)
        sim_neg = (anchor * negative).sum(dim=1)
        
        # Weight factors
        alpha_p = F.relu(self.O_p - sim_pos.detach())
        alpha_n = F.relu(sim_neg.detach() - self.O_n)
        
        # Compute loss
        logit_p = -self.gamma * alpha_p * (sim_pos - self.Delta_p)
        logit_n = self.gamma * alpha_n * (sim_neg - self.Delta_n)
        
        loss = F.softplus(logit_p + logit_n)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AnglELoss(nn.Module):
    """AnglE (Angle Embedding) loss.
    
    Uses angle difference in complex space for better gradient properties.
    
    Reference: AnglE: Optimizing Text Embeddings with Prompt Optimization
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        reduction: str = "mean",
    ):
        """Initialize AnglE loss.
        
        Args:
            temperature: Temperature scaling
            reduction: Reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
        logger.debug(f"AnglELoss initialized: temperature={temperature}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """Compute AnglE loss.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss tensor
        """
        batch_size = anchor.shape[0]
        embedding_dim = anchor.shape[1]
        
        # Split embeddings into real and imaginary parts
        half_dim = embedding_dim // 2
        anchor_real = anchor[:, :half_dim]
        anchor_imag = anchor[:, half_dim:2*half_dim]
        positive_real = positive[:, :half_dim]
        positive_imag = positive[:, half_dim:2*half_dim]
        
        # Compute angle-based similarity
        # Re(z1 * conj(z2)) = real1*real2 + imag1*imag2
        # Im(z1 * conj(z2)) = imag1*real2 - real1*imag2
        
        # For all pairs
        real_part = torch.matmul(anchor_real, positive_real.T) + torch.matmul(anchor_imag, positive_imag.T)
        imag_part = torch.matmul(anchor_imag, positive_real.T) - torch.matmul(anchor_real, positive_imag.T)
        
        # Angle similarity (using abs for magnitude)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2 + 1e-8)
        similarity = magnitude / self.temperature
        
        # InfoNCE-style loss
        labels = torch.arange(batch_size, device=anchor.device)
        loss = F.cross_entropy(similarity, labels, reduction=self.reduction)
        
        return loss

