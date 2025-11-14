"""
Recommendation Module
Provides similarity search and music recommendations
"""

from .similarity_search import (
    SimilaritySearcher,
    FAISSSearcher
)
from .recommender import MusicRecommender

__all__ = [
    'SimilaritySearcher',
    'FAISSSearcher',
    'MusicRecommender'
]
