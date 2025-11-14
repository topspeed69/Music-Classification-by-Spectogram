"""
Contrastive Loss Functions
Implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    Used in SimCLR for contrastive learning.
    
    Reference:
        Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations"
        https://arxiv.org/abs/2002.05709
    """
    
    def __init__(self, temperature: float = 0.5, use_cosine_similarity: bool = True):
        """
        Args:
            temperature: Temperature parameter for scaling
            use_cosine_similarity: If True, use cosine similarity; else dot product
        """
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.similarity_fn = self._cosine_similarity if use_cosine_similarity else self._dot_product
        
    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between all pairs.
        
        Args:
            x: Embeddings [N, D]
            y: Embeddings [N, D]
        Returns:
            Similarity matrix [N, N]
        """
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return torch.mm(x, y.t())
    
    def _dot_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute dot product between all pairs.
        
        Args:
            x: Embeddings [N, D]
            y: Embeddings [N, D]
        Returns:
            Similarity matrix [N, N]
        """
        return torch.mm(x, y.t())
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z_i: Embeddings from first augmentation [B, D]
            z_j: Embeddings from second augmentation [B, D]
        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]
        
        # Combine embeddings
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        similarity_matrix = self.similarity_fn(representations, representations)  # [2B, 2B]
        
        # Create mask for positive pairs
        # Positive pairs: (i, i+B) and (i+B, i) for i in [0, B)
        mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        positives_mask = mask.repeat(2, 2)  # [2B, 2B]
        
        # Create mask for negative pairs (all except self and positive pair)
        negatives_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        
        # Extract positive and negative similarities
        # For each sample, positive is at distance B
        pos_indices = torch.arange(2 * batch_size, device=z_i.device)
        pos_indices = torch.cat([pos_indices[batch_size:], pos_indices[:batch_size]])
        
        # Positive similarity: similarity with augmented version
        positives = similarity_matrix[torch.arange(2 * batch_size), pos_indices].reshape(2 * batch_size, 1)
        
        # Negative similarities: all other samples
        negatives = similarity_matrix[negatives_mask].reshape(2 * batch_size, -1)
        
        # Compute logits
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        
        # Labels: positive pair is at index 0
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (alternative formulation of contrastive loss).
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            z_i: Embeddings from first augmentation [B, D]
            z_j: Embeddings from second augmentation [B, D]
        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        # Positive pairs: cosine similarity
        pos_sim = torch.sum(z_i * z_j, dim=1) / self.temperature  # [B]
        
        # Negative pairs: similarity with all other samples in batch
        # Compute similarity matrix
        sim_matrix = torch.mm(z_i, z_j.t()) / self.temperature  # [B, B]
        
        # For each sample, exclude self-similarity
        # Log-sum-exp over all samples (including positive)
        exp_sim = torch.exp(sim_matrix)
        
        # Sum over all samples (denominators)
        sum_exp = exp_sim.sum(dim=1)
        
        # Loss: -log(exp(pos) / sum(exp(all)))
        loss = -torch.log(torch.exp(pos_sim) / sum_exp)
        
        return loss.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (can be used if labels are available).
    Reference: Khosla et al. "Supervised Contrastive Learning"
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Embeddings [2B, D] (two views concatenated)
            labels: Labels [B] (optional, for supervised version)
        Returns:
            Scalar loss value
        """
        batch_size = features.shape[0] // 2
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.t())  # [2B, 2B]
        
        # If labels provided, use supervised loss
        if labels is not None:
            # Create mask for same-label pairs
            labels = labels.repeat(2)  # [2B]
            mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [2B, 2B]
            
            # Remove self-similarity
            mask.fill_diagonal_(False)
        else:
            # Self-supervised: positive pairs are augmentations of same sample
            mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
            for i in range(batch_size):
                mask[i, i + batch_size] = True
                mask[i + batch_size, i] = True
        
        # Compute loss
        similarity_matrix = similarity_matrix / self.temperature
        exp_sim = torch.exp(similarity_matrix)
        
        # For each anchor, sum exp similarities
        sum_exp = exp_sim.sum(dim=1, keepdim=True)
        
        # Compute log probability for positive pairs
        log_prob = similarity_matrix - torch.log(sum_exp)
        
        # Mean over positive pairs
        mask_sum = mask.sum(dim=1)
        mask_sum[mask_sum == 0] = 1  # Avoid division by zero
        
        loss = -(log_prob * mask).sum(dim=1) / mask_sum
        
        return loss.mean()


def get_contrastive_loss(loss_type: str = 'nt_xent', temperature: float = 0.5, 
                         use_cosine_similarity: bool = True) -> nn.Module:
    """
    Factory function to get contrastive loss.
    
    Args:
        loss_type: Type of loss ('nt_xent', 'infonce', 'supcon')
        temperature: Temperature parameter
        use_cosine_similarity: Use cosine similarity (for nt_xent)
    Returns:
        Loss module
    """
    if loss_type == 'nt_xent':
        return NTXentLoss(temperature=temperature, use_cosine_similarity=use_cosine_similarity)
    elif loss_type == 'infonce':
        return InfoNCELoss(temperature=temperature)
    elif loss_type == 'supcon':
        return SupConLoss(temperature=temperature)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
