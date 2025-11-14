"""
Embedding Extraction Module
Extract and store embeddings from trained model
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import yaml
import argparse
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from CNN.models import build_model
from CNN.augmentation import get_augmentation_pipeline
from CNN.data import create_inference_dataloader


def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings for all samples in dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader for spectrograms
        device: Device to use
    Returns:
        (embeddings, paths) - numpy arrays
    """
    model.eval()
    
    all_embeddings = []
    all_paths = []
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            
            # Get embeddings
            embeddings = model.get_embeddings(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_paths.extend(paths)
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    
    return embeddings, all_paths


def save_embeddings(embeddings, paths, output_path):
    """
    Save embeddings and paths to disk.
    
    Args:
        embeddings: Numpy array [N, D]
        paths: List of file paths
        output_path: Output file path
    """
    data = {
        'embeddings': embeddings,
        'paths': paths,
        'shape': embeddings.shape
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved {len(paths)} embeddings to {output_path}")
    print(f"Embedding shape: {embeddings.shape}")


def load_embeddings(embeddings_path):
    """
    Load embeddings from disk.
    
    Args:
        embeddings_path: Path to embeddings file
    Returns:
        (embeddings, paths) - numpy array and list of paths
    """
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['embeddings'], data['paths']


def main(args):
    """Main function for embedding extraction"""
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create augmentation pipeline (inference mode - no augmentation)
    transform = get_augmentation_pipeline(config, training=False)
    
    # Create dataloader
    print(f"Loading spectrograms from {args.data_dir}")
    dataloader = create_inference_dataloader(
        data_dir=args.data_dir,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, paths = extract_embeddings(model, dataloader, device)
    
    # Save embeddings
    output_path = Path(args.output_dir) / 'embeddings.pkl'
    save_embeddings(embeddings, paths, output_path)
    
    print("\nDone!")
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"Number of files: {len(paths)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing spectrograms')
    parser.add_argument('--output_dir', type=str, default='embeddings_db',
                       help='Output directory for embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)
