"""
Embeddings Module
Handles embedding extraction and storage
"""

from .extract_embeddings import (
    extract_embeddings,
    save_embeddings,
    load_embeddings
)

__all__ = [
    'extract_embeddings',
    'save_embeddings',
    'load_embeddings'
]
