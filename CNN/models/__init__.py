"""
Models Module
Contains CNN encoder and projection head for self-supervised learning
"""

from .encoder import (
    AudioEncoderCNN,
    ProjectionHead,
    ContrastiveModel,
    build_model
)

__all__ = [
    'AudioEncoderCNN',
    'ProjectionHead',
    'ContrastiveModel',
    'build_model'
]
