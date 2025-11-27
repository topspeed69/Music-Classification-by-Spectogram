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

from .audio_augmentations import (
    ContrastiveAudioAugmentation,
    AudioDatasetWithAugmentation,
    PitchShift,
    TempoStretch,
    GainAdjustment,
    ParametricEQ,
    DynamicRangeCompression,
    AddNoise,
    AddReverb
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
    'get_augmentation_pipeline',
    'ContrastiveAudioAugmentation',
    'AudioDatasetWithAugmentation',
    'PitchShift',
    'TempoStretch',
    'GainAdjustment',
    'ParametricEQ',
    'DynamicRangeCompression',
    'AddNoise',
    'AddReverb'
]
