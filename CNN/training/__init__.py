"""
Training Module
Handles model training with contrastive learning
"""

from .contrastive_loss import (
    NTXentLoss,
    InfoNCELoss,
    SupConLoss,
    get_contrastive_loss
)

__all__ = [
    'NTXentLoss',
    'InfoNCELoss',
    'SupConLoss',
    'get_contrastive_loss'
]
