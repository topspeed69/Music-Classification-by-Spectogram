"""
Augmentation Module
Provides data augmentation for self-supervised learning on spectrograms
"""

from .augmentations import (
    RandomCrop,
    ColorJitter,
    GaussianNoise,
    HorizontalFlip,
    FrequencyMasking,
    TimeMasking,
    RandomAugmentation,
    SpectrogramAugmentation,
    get_augmentation_pipeline
)

__all__ = [
    'RandomCrop',
    'ColorJitter',
    'GaussianNoise',
    'HorizontalFlip',
    'FrequencyMasking',
    'TimeMasking',
    'RandomAugmentation',
    'SpectrogramAugmentation',
    'get_augmentation_pipeline'
]
