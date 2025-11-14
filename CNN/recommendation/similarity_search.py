"""
Similarity Search Module
Fast cosine similarity search for song recommendations
"""

import numpy as np
import torch
from typing import List, Tuple


class SimilaritySearcher:
    """
    Efficient similarity search using cosine similarity.
    """
    
    def __init__(self, embeddings: np.ndarray, paths: List[str]):
        """
        Args:
            embeddings: Database embeddings [N, D]
            paths: List of file paths corresponding to embeddings
        """
        self.embeddings = embeddings
        self.paths = paths
        self.num_embeddings = len(embeddings)
        
        # Normalize embeddings for fast cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = embeddings / (norms + 1e-8)
        
        print(f"Initialized similarity searcher with {self.num_embeddings} embeddings")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               exclude_self: bool = True) -> Tuple[List[str], np.ndarray]:
        """
        Find most similar songs to query.
        
        Args:
            query_embedding: Query embedding [D] or [1, D]
            top_k: Number of top results to return
            exclude_self: If True, exclude exact matches (for when query is in database)
        Returns:
            (paths, similarities) - lists of file paths and similarity scores
        """
        # Normalize query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_norm = np.linalg.norm(query_embedding)
        query_normalized = query_embedding / (query_norm + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(self.normalized_embeddings, query_normalized.T).squeeze()
        
        # Get top-k indices
        if exclude_self:
            # Exclude very high similarities (likely exact matches)
            similarities[similarities > 0.9999] = -1
        
        top_k = min(top_k, self.num_embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        # Get corresponding paths
        top_paths = [self.paths[idx] for idx in top_indices]
        
        return top_paths, top_similarities
    
    def batch_search(self, query_embeddings: np.ndarray, top_k: int = 10) -> List[Tuple[List[str], np.ndarray]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: Query embeddings [M, D]
            top_k: Number of top results per query
        Returns:
            List of (paths, similarities) for each query
        """
        results = []
        for query_emb in query_embeddings:
            paths, sims = self.search(query_emb, top_k=top_k, exclude_self=False)
            results.append((paths, sims))
        
        return results
    
    def get_statistics(self) -> dict:
        """Get statistics about the embedding database"""
        return {
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embeddings.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(self.embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(self.embeddings, axis=1)))
        }


class FAISSSearcher:
    """
    Fast approximate nearest neighbor search using FAISS.
    Requires: pip install faiss-cpu (or faiss-gpu)
    """
    
    def __init__(self, embeddings: np.ndarray, paths: List[str], use_gpu: bool = False):
        """
        Args:
            embeddings: Database embeddings [N, D]
            paths: List of file paths
            use_gpu: Whether to use GPU for FAISS
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        self.embeddings = embeddings.astype(np.float32)
        self.paths = paths
        self.dimension = embeddings.shape[1]
        self.num_embeddings = len(embeddings)
        
        # Normalize embeddings
        faiss.normalize_L2(self.embeddings)
        
        # Create index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        
        print(f"Initialized FAISS index with {self.num_embeddings} embeddings")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[List[str], np.ndarray]:
        """
        Fast similarity search using FAISS.
        
        Args:
            query_embedding: Query embedding [D] or [1, D]
            top_k: Number of results
        Returns:
            (paths, similarities)
        """
        import faiss
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        top_k = min(top_k, self.num_embeddings)
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Get paths
        top_paths = [self.paths[idx] for idx in indices[0]]
        
        return top_paths, similarities[0]
