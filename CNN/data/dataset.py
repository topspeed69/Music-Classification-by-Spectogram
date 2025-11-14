"""
Dataset and DataLoader for Audio Spectrograms
Handles loading spectrograms and applying augmentations
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Callable
import random


class SpectrogramDataset(Dataset):
    """
    Dataset for loading spectrogram images.
    Returns two augmented views of each spectrogram for contrastive learning.
    """
    
    def __init__(self, data_dir: str, transform: Optional[Callable] = None, 
                 return_pair: bool = True):
        """
        Args:
            data_dir: Directory containing spectrogram images
            transform: Transformation/augmentation to apply
            return_pair: If True, return two augmented views; else return single view
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_pair = return_pair
        
        # Get all image files
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Found {len(self.image_paths)} spectrograms in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and transform spectrogram.
        
        Args:
            idx: Sample index
        Returns:
            If return_pair: (view1, view2) - two augmented views
            Else: single view
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        if self.return_pair:
            # Apply two different augmentations
            if self.transform:
                view1 = self.transform(image)
                view2 = self.transform(image)
            else:
                view1 = image
                view2 = image
            
            return view1, view2
        else:
            # Single view (for validation/test)
            if self.transform:
                image = self.transform(image)
            
            return image, str(img_path)
    
    def get_image_path(self, idx: int) -> str:
        """Get image path for a given index"""
        return str(self.image_paths[idx])


class AudioFileDataset(Dataset):
    """
    Dataset for loading audio files directly.
    Can be used with spectrogram conversion on-the-fly.
    """
    
    def __init__(self, audio_dir: str, file_list: Optional[str] = None, 
                 transform: Optional[Callable] = None, return_pair: bool = True):
        """
        Args:
            audio_dir: Directory containing audio files
            file_list: Optional text file with list of audio files
            transform: Transformation/augmentation to apply
            return_pair: If True, return two augmented views
        """
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.return_pair = return_pair
        
        # Get audio files
        if file_list:
            with open(file_list, 'r') as f:
                self.audio_paths = [self.audio_dir / line.strip() for line in f]
        else:
            self.audio_paths = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                self.audio_paths.extend(list(self.audio_dir.rglob(ext)))
        
        if len(self.audio_paths) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        print(f"Found {len(self.audio_paths)} audio files in {audio_dir}")
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int):
        """
        Load audio file (implement spectrogram conversion here if needed).
        For now, this is a placeholder.
        """
        audio_path = self.audio_paths[idx]
        # TODO: Implement audio loading and spectrogram conversion
        raise NotImplementedError("Audio loading not implemented. Use pre-computed spectrograms.")


def create_dataloaders(config: dict, train_transform: Callable, 
                       val_transform: Callable) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        train_transform: Augmentation for training
        val_transform: Transformation for validation (no augmentation)
    Returns:
        (train_loader, val_loader)
    """
    data_config = config.get('data', {})
    paths_config = config.get('paths', {}) if 'paths' in config else config.get('training', {}).get('paths', {})
    
    data_dir = paths_config.get('data_dir', 'AudioToSpectogram/output_mel')
    batch_size = data_config.get('dataloader', {}).get('batch_size', 128)
    num_workers = data_config.get('dataloader', {}).get('num_workers', 4)
    pin_memory = data_config.get('dataloader', {}).get('pin_memory', True)
    
    # Create full dataset
    full_dataset = SpectrogramDataset(
        data_dir=data_dir,
        transform=train_transform,
        return_pair=True
    )
    
    # Split into train and validation
    split_config = data_config.get('split', {})
    train_ratio = split_config.get('train', 0.8)
    
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('training', {}).get('seed', 42))
    )
    
    # Update validation dataset to use val transform
    val_dataset_obj = SpectrogramDataset(
        data_dir=data_dir,
        transform=val_transform,
        return_pair=True
    )
    val_dataset.dataset = val_dataset_obj
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_inference_dataloader(data_dir: str, transform: Callable, 
                                batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    """
    Create dataloader for inference (embedding extraction).
    
    Args:
        data_dir: Directory containing spectrograms
        transform: Transformation to apply
        batch_size: Batch size
        num_workers: Number of workers
    Returns:
        DataLoader for inference
    """
    dataset = SpectrogramDataset(
        data_dir=data_dir,
        transform=transform,
        return_pair=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return loader
