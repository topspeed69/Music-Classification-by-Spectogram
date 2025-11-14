"""
Music Recommender System
Provides song recommendations based on audio similarity
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from CNN.models import build_model
from CNN.augmentation import get_augmentation_pipeline
from CNN.embeddings import load_embeddings
from CNN.recommendation.similarity_search import SimilaritySearcher


class MusicRecommender:
    """
    Complete music recommendation system.
    Combines model inference with similarity search.
    """
    
    def __init__(self, model_path: str, embeddings_db: str, use_faiss: bool = False):
        """
        Args:
            model_path: Path to trained model checkpoint
            embeddings_db: Path to embeddings database file
            use_faiss: Whether to use FAISS for fast search
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        self.model = build_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load transform (inference mode)
        self.transform = get_augmentation_pipeline(self.config, training=False)
        
        # Load embeddings database
        print(f"Loading embeddings from {embeddings_db}")
        embeddings, paths = load_embeddings(embeddings_db)
        
        # Initialize searcher
        if use_faiss:
            try:
                from CNN.recommendation.similarity_search import FAISSSearcher
                self.searcher = FAISSSearcher(embeddings, paths)
            except ImportError:
                print("FAISS not available, using standard searcher")
                self.searcher = SimilaritySearcher(embeddings, paths)
        else:
            self.searcher = SimilaritySearcher(embeddings, paths)
        
        print("Recommender initialized successfully!")
    
    def get_embedding_from_spectrogram(self, spectrogram_path: str) -> np.ndarray:
        """
        Extract embedding from a spectrogram image.
        
        Args:
            spectrogram_path: Path to spectrogram image
        Returns:
            Embedding vector [D]
        """
        # Load image
        image = Image.open(spectrogram_path).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Apply transform
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model.get_embeddings(image)
        
        return embedding.cpu().numpy().squeeze()
    
    def recommend_from_spectrogram(self, spectrogram_path: str, top_k: int = 10, 
                                   exclude_self: bool = True) -> list:
        """
        Get song recommendations from a spectrogram.
        
        Args:
            spectrogram_path: Path to query spectrogram
            top_k: Number of recommendations
            exclude_self: Exclude exact matches
        Returns:
            List of (path, similarity_score) tuples
        """
        # Get embedding
        embedding = self.get_embedding_from_spectrogram(spectrogram_path)
        
        # Search
        paths, similarities = self.searcher.search(embedding, top_k=top_k, exclude_self=exclude_self)
        
        # Format results
        recommendations = [(path, float(sim)) for path, sim in zip(paths, similarities)]
        
        return recommendations
    
    def recommend_from_audio(self, audio_path: str, top_k: int = 10) -> list:
        """
        Get recommendations directly from audio file.
        NOTE: Requires audio-to-spectrogram conversion.
        
        Args:
            audio_path: Path to audio file
            top_k: Number of recommendations
        Returns:
            List of (path, similarity_score) tuples
        """
        # TODO: Implement audio-to-spectrogram conversion on-the-fly
        raise NotImplementedError("Direct audio recommendation not yet implemented. "
                                "Convert to spectrogram first using audio_to_spectogram_mel.py")
    
    def batch_recommend(self, spectrogram_paths: list, top_k: int = 10) -> list:
        """
        Get recommendations for multiple spectrograms.
        
        Args:
            spectrogram_paths: List of spectrogram paths
            top_k: Number of recommendations per query
        Returns:
            List of recommendation lists
        """
        all_recommendations = []
        
        for spec_path in spectrogram_paths:
            recommendations = self.recommend_from_spectrogram(spec_path, top_k=top_k)
            all_recommendations.append(recommendations)
        
        return all_recommendations
    
    def get_database_stats(self) -> dict:
        """Get statistics about the embeddings database"""
        return self.searcher.get_statistics()


def main():
    """Demo: Command-line interface for recommendations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Music Recommendation System')
    parser.add_argument('--query_song', type=str, required=True,
                       help='Path to query spectrogram')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--embeddings_db', type=str, required=True,
                       help='Path to embeddings database')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of recommendations')
    parser.add_argument('--use_faiss', action='store_true',
                       help='Use FAISS for fast search')
    
    args = parser.parse_args()
    
    # Initialize recommender
    recommender = MusicRecommender(
        model_path=args.model_checkpoint,
        embeddings_db=args.embeddings_db,
        use_faiss=args.use_faiss
    )
    
    # Get recommendations
    print(f"\nGetting recommendations for: {args.query_song}")
    recommendations = recommender.recommend_from_spectrogram(
        args.query_song,
        top_k=args.top_k
    )
    
    # Display results
    print(f"\nTop {args.top_k} recommendations:")
    print("-" * 80)
    for rank, (path, similarity) in enumerate(recommendations, 1):
        print(f"{rank:2d}. {Path(path).name:<50} (similarity: {similarity:.4f})")
    
    # Show database stats
    print("\n" + "=" * 80)
    stats = recommender.get_database_stats()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
