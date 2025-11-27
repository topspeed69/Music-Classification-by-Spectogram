"""
Integration script for ContrastiveAudioAugmentation with existing training pipeline
Shows how to use the new audio augmentation pipeline with the current CNN training setup
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from augmentation.audio_augmentations import (
    ContrastiveAudioAugmentation,
    AudioDatasetWithAugmentation
)
from models.encoder import AudioEncoder
from training.contrastive_loss import ContrastiveLoss
from training.train import train_epoch, validate_epoch


def setup_audio_augmentation_pipeline(config_path: str = None):
    """
    Setup audio augmentation pipeline with configuration
    
    Args:
        config_path: Path to YAML config file (optional)
    Returns:
        Augmentation pipeline instance
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract augmentation config
        aug_config = config.get('augmentation', {})
        audio_config = aug_config.get('audio', {})
        
        # Initialize with config
        aug_pipeline = ContrastiveAudioAugmentation(
            sr=audio_config.get('sample_rate', 22050),
            n_fft=audio_config.get('n_fft', 2048),
            hop_length=audio_config.get('hop_length', 512),
            n_mels=audio_config.get('n_mels', 128),
            duration=audio_config.get('duration', 3.0),
            num_waveform_augs=audio_config.get('num_waveform_augs', 3),
            num_spectrogram_augs=audio_config.get('num_spectrogram_augs', 2),
            max_workers=audio_config.get('max_workers', 7),
            waveform_aug_probs=audio_config.get('waveform_probs'),
            spectrogram_aug_probs=audio_config.get('spectrogram_probs')
        )
    else:
        # Use default configuration
        aug_pipeline = ContrastiveAudioAugmentation(
            sr=22050,
            n_mels=128,
            duration=3.0,
            num_waveform_augs=3,
            num_spectrogram_augs=2,
            max_workers=7
        )
    
    print("✓ Audio augmentation pipeline initialized")
    return aug_pipeline


def create_dataloaders(
    train_audio_dir: str,
    val_audio_dir: str,
    augmentation: ContrastiveAudioAugmentation,
    batch_size: int = 32,
    num_workers: int = 5,
    train_val_split: float = None
):
    """
    Create training and validation dataloaders with audio augmentation
    
    Args:
        train_audio_dir: Path to training audio directory
        val_audio_dir: Path to validation audio directory (can be same as train_dir if using split)
        augmentation: Augmentation pipeline instance
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        train_val_split: If provided (0-1), automatically split dataset (e.g., 0.8 for 80% train)
    Returns:
        train_loader, val_loader
    """
    if train_val_split is not None:
        # Use same directory and split automatically
        from sklearn.model_selection import train_test_split
        
        full_dataset = AudioDatasetWithAugmentation(
            audio_dir=train_audio_dir,
            augmentation=augmentation,
            file_extensions=('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        )
        
        # Split indices
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(
            indices, train_size=train_val_split, random_state=42
        )
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        print(f"✓ Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")
        
    else:
        # Use separate directories
        train_dataset = AudioDatasetWithAugmentation(
            audio_dir=train_audio_dir,
            augmentation=augmentation,
            file_extensions=('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        )
        
        val_dataset = AudioDatasetWithAugmentation(
            audio_dir=val_audio_dir,
            augmentation=augmentation,
            file_extensions=('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        )
        
        print(f"✓ Training dataset: {len(train_dataset)} samples")
        print(f"✓ Validation dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def train_with_audio_augmentation(
    train_audio_dir: str = "../../AudioToSpectogram/fma_small_dataset",
    val_audio_dir: str = None,
    train_val_split: float = 0.8,
    config_path: str = None,
    output_dir: str = "../../checkpoints",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = None
):
    """
    Complete training pipeline with audio augmentation
    
    Args:
        train_audio_dir: Path to training audio files (or full dataset if using split)
        val_audio_dir: Path to validation audio files (None if using automatic split)
        train_val_split: Proportion of data for training (0-1), e.g., 0.8 for 80% train, 20% val
        config_path: Path to configuration file
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device to use ('cuda' or 'cpu')
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup augmentation pipeline
    aug_pipeline = setup_audio_augmentation_pipeline(config_path)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_audio_dir=train_audio_dir,
        val_audio_dir=val_audio_dir if val_audio_dir else train_audio_dir,
        augmentation=aug_pipeline,
        batch_size=batch_size,
        num_workers=4,
        train_val_split=train_val_split if not val_audio_dir else None
    )
    
    # Initialize model
    model = AudioEncoder(
        input_channels=1,  # Mono audio spectrograms
        embedding_dim=256,
        base_channels=64
    ).to(device)
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(temperature=0.07)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("Starting training with audio augmentation pipeline")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_idx, (views1, views2) in enumerate(train_loader):
            # Move to device
            views1 = views1.to(device)
            views2 = views2.to(device)
            
            # Forward pass
            embeddings1 = model(views1)
            embeddings2 = model(views2)
            
            # Compute loss
            loss = criterion(embeddings1, embeddings2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for views1, views2 in val_loader:
                views1 = views1.to(device)
                views2 = views2.to(device)
                
                embeddings1 = model(views1)
                embeddings2 = model(views2)
                
                loss = criterion(embeddings1, embeddings2)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_path / f"checkpoint_audio_aug_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = output_path / "best_model_audio_aug.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"  ✓ Best model saved: {best_model_path} (Val Loss: {best_val_loss:.4f})")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with audio augmentation pipeline')
    parser.add_argument('--train-dir', type=str, 
                       default='../../AudioToSpectogram/fma_small_dataset',
                       help='Path to training audio directory (or full dataset if using --split)')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='Path to validation audio directory (optional if using --split)')
    parser.add_argument('--split', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8 = 80%% train, 20%% val)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='../../checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not Path(args.train_dir).exists():
        print(f"Error: Training directory not found: {args.train_dir}")
        print("\nPlease ensure you have the fma_small_dataset in the AudioToSpectogram folder")
        print("Expected structure:")
        print("  AudioToSpectogram/")
        print("    fma_small_dataset/")
        print("      000002.mp3")
        print("      000005.mp3")
        print("      ... (all audio files directly here)")
        return
    
    # Start training
    train_with_audio_augmentation(
        train_audio_dir=args.train_dir,
        val_audio_dir=args.val_dir,
        train_val_split=args.split if not args.val_dir else None,
        config_path=args.config,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
