"""
Data Augmentation Module for Audio Spectrograms
Implements 5+ augmentation techniques for self-supervised learning
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import random
import numpy as np
from typing import List, Tuple, Optional


class RandomCrop(nn.Module):
    """Randomly crop spectrogram to learn position invariance"""
    
    def __init__(self, crop_size: Tuple[int, int], scale: Tuple[float, float] = (0.7, 1.0)):
        super().__init__()
        self.crop_size = crop_size
        self.scale = scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [C, H, W]
        Returns:
            Cropped tensor [C, crop_h, crop_w]
        """
        _, h, w = x.shape
        
        # Random scale
        scale_factor = random.uniform(*self.scale)
        crop_h = int(self.crop_size[0] * scale_factor)
        crop_w = int(self.crop_size[1] * scale_factor)
        
        # Ensure crop size doesn't exceed image size
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        # Random position
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        cropped = x[:, top:top+crop_h, left:left+crop_w]
        
        # Resize back to original crop size
        cropped = T.functional.resize(cropped, self.crop_size, antialias=True)
        
        return cropped


class ColorJitter(nn.Module):
    """Simulate intensity variations in spectrogram"""
    
    def __init__(self, brightness: float = 0.4, contrast: float = 0.4, 
                 saturation: float = 0.4, hue: float = 0.1):
        super().__init__()
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [C, H, W]
        Returns:
            Color jittered tensor [C, H, W]
        """
        return self.jitter(x)


class GaussianNoise(nn.Module):
    """Add Gaussian noise for robustness"""
    
    def __init__(self, mean: float = 0.0, std: float = 0.05, noise_type: str = 'additive'):
        super().__init__()
        self.mean = mean
        self.std = std
        self.noise_type = noise_type
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [C, H, W]
        Returns:
            Noisy tensor [C, H, W]
        """
        noise = torch.randn_like(x) * self.std + self.mean
        
        if self.noise_type == 'additive':
            return x + noise
        elif self.noise_type == 'multiplicative':
            return x * (1 + noise)
        else:
            return x


class HorizontalFlip(nn.Module):
    """Flip spectrogram horizontally (time reversal)"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [C, H, W]
        Returns:
            Flipped tensor [C, H, W]
        """
        if random.random() < self.p:
            return torch.flip(x, dims=[-1])  # Flip along width (time axis)
        return x


class FrequencyMasking(nn.Module):
    """Mask random frequency bands (SpecAugment)"""
    
    def __init__(self, max_mask_percentage: float = 0.15, num_masks: int = 2):
        super().__init__()
        self.max_mask_percentage = max_mask_percentage
        self.num_masks = num_masks
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [C, H, W]
        Returns:
            Masked tensor [C, H, W]
        """
        cloned = x.clone()
        _, h, _ = cloned.shape
        
        for _ in range(self.num_masks):
            # Random mask size
            mask_size = int(h * random.uniform(0, self.max_mask_percentage))
            if mask_size == 0:
                continue
                
            # Random position
            mask_start = random.randint(0, h - mask_size)
            
            # Apply mask (set to mean value)
            cloned[:, mask_start:mask_start+mask_size, :] = cloned.mean()
            
        return cloned


class TimeMasking(nn.Module):
    """Mask random time steps (SpecAugment)"""
    
    def __init__(self, max_mask_percentage: float = 0.15, num_masks: int = 2):
        super().__init__()
        self.max_mask_percentage = max_mask_percentage
        self.num_masks = num_masks
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [C, H, W]
        Returns:
            Masked tensor [C, H, W]
        """
        cloned = x.clone()
        _, _, w = cloned.shape
        
        for _ in range(self.num_masks):
            # Random mask size
            mask_size = int(w * random.uniform(0, self.max_mask_percentage))
            if mask_size == 0:
                continue
                
            # Random position
            mask_start = random.randint(0, w - mask_size)
            
            # Apply mask (set to mean value)
            cloned[:, :, mask_start:mask_start+mask_size] = cloned.mean()
            
        return cloned


class RandomAugmentation(nn.Module):
    """
    Randomly apply N augmentations from a pool of transforms.
    This implements the augmentation strategy for self-supervised learning.
    """
    
    def __init__(self, augmentations: List[nn.Module], num_augmentations: int = 3, 
                 apply_prob: float = 0.8):
        """
        Args:
            augmentations: List of augmentation modules
            num_augmentations: Number of augmentations to apply per sample
            apply_prob: Probability to apply each selected augmentation
        """
        super().__init__()
        self.augmentations = augmentations
        self.num_augmentations = min(num_augmentations, len(augmentations))
        self.apply_prob = apply_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [C, H, W]
        Returns:
            Augmented tensor [C, H, W]
        """
        # Randomly select augmentations
        selected_augs = random.sample(self.augmentations, self.num_augmentations)
        
        # Apply selected augmentations
        for aug in selected_augs:
            if random.random() < self.apply_prob:
                x = aug(x)
                
        return x


class SpectrogramAugmentation(nn.Module):
    """
    Complete augmentation pipeline for spectrograms.
    Applies normalization and random augmentations.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary with augmentation parameters
        """
        super().__init__()
        
        aug_config = config.get('augmentation', {})
        transforms_config = aug_config.get('transforms', {})
        
        # Resize configuration (to ensure consistent tensor sizes)
        resize_config = aug_config.get('resize', {})
        if resize_config.get('enabled', True):
            height = resize_config.get('height', 128)
            width = resize_config.get('width', 1293)
            interpolation_mode = resize_config.get('interpolation', 'bilinear')
            
            # Map interpolation string to torchvision mode
            interp_map = {
                'bilinear': T.InterpolationMode.BILINEAR,
                'bicubic': T.InterpolationMode.BICUBIC,
                'nearest': T.InterpolationMode.NEAREST
            }
            interp = interp_map.get(interpolation_mode, T.InterpolationMode.BILINEAR)
            
            self.resize = T.Resize((height, width), interpolation=interp, antialias=True)
        else:
            self.resize = None
        
        # Build augmentation list
        augmentations = []
        
        # 1. Random Crop - Disabled by default for consistent CNN input sizes
        if transforms_config.get('random_crop', {}).get('enabled', False):
            crop_cfg = transforms_config['random_crop']
            augmentations.append(
                RandomCrop(
                    crop_size=crop_cfg.get('crop_size', [96, 970]),
                    scale=crop_cfg.get('scale', [0.7, 1.0])
                )
            )
        
        # 2. Color Jitter
        if transforms_config.get('color_jitter', {}).get('enabled', True):
            jitter_cfg = transforms_config['color_jitter']
            augmentations.append(
                ColorJitter(
                    brightness=jitter_cfg.get('brightness', 0.4),
                    contrast=jitter_cfg.get('contrast', 0.4),
                    saturation=jitter_cfg.get('saturation', 0.4),
                    hue=jitter_cfg.get('hue', 0.1)
                )
            )
        
        # 3. Gaussian Noise
        if transforms_config.get('gaussian_noise', {}).get('enabled', True):
            noise_cfg = transforms_config['gaussian_noise']
            augmentations.append(
                GaussianNoise(
                    mean=noise_cfg.get('mean', 0.0),
                    std=noise_cfg.get('std', 0.05),
                    noise_type=noise_cfg.get('noise_type', 'additive')
                )
            )
        
        # 4. Horizontal Flip
        if transforms_config.get('horizontal_flip', {}).get('enabled', True):
            flip_cfg = transforms_config['horizontal_flip']
            augmentations.append(
                HorizontalFlip(p=flip_cfg.get('probability', 0.5))
            )
        
        # 5. Frequency Masking
        if transforms_config.get('frequency_masking', {}).get('enabled', True):
            freq_cfg = transforms_config['frequency_masking']
            augmentations.append(
                FrequencyMasking(
                    max_mask_percentage=freq_cfg.get('max_mask_percentage', 0.15),
                    num_masks=freq_cfg.get('num_masks', 2)
                )
            )
        
        # 6. Time Masking
        if transforms_config.get('time_masking', {}).get('enabled', True):
            time_cfg = transforms_config['time_masking']
            augmentations.append(
                TimeMasking(
                    max_mask_percentage=time_cfg.get('max_mask_percentage', 0.15),
                    num_masks=time_cfg.get('num_masks', 2)
                )
            )
        
        # Random augmentation selector
        self.random_aug = RandomAugmentation(
            augmentations=augmentations,
            num_augmentations=aug_config.get('num_augmentations', 3),
            apply_prob=aug_config.get('apply_probability', 0.8)
        )
        
        # Normalization
        norm_config = aug_config.get('normalization', {})
        self.normalize = T.Normalize(
            mean=norm_config.get('mean', [0.485, 0.456, 0.406]),
            std=norm_config.get('std', [0.229, 0.224, 0.225])
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrogram tensor [C, H, W]
        Returns:
            Augmented and normalized tensor [C, H, W]
        """
        # First, resize to ensure consistent dimensions across all samples
        if self.resize is not None:
            x = self.resize(x)
        
        # Apply random augmentations
        x = self.random_aug(x)
        
        # Normalize
        x = self.normalize(x)
        
        return x


def get_augmentation_pipeline(config: dict, training: bool = True) -> nn.Module:
    """
    Factory function to create augmentation pipeline.
    
    Args:
        config: Configuration dictionary
        training: If True, apply augmentations; if False, only normalize
    
    Returns:
        Augmentation module
    """
    if training:
        return SpectrogramAugmentation(config)
    else:
        # Validation/test: resize and normalize only
        aug_config = config.get('augmentation', {})
        resize_config = aug_config.get('resize', {})
        norm_config = aug_config.get('normalization', {})
        
        transforms = []
        
        # Add resize if enabled
        if resize_config.get('enabled', True):
            height = resize_config.get('height', 128)
            width = resize_config.get('width', 1293)
            interpolation_mode = resize_config.get('interpolation', 'bilinear')
            
            interp_map = {
                'bilinear': T.InterpolationMode.BILINEAR,
                'bicubic': T.InterpolationMode.BICUBIC,
                'nearest': T.InterpolationMode.NEAREST
            }
            interp = interp_map.get(interpolation_mode, T.InterpolationMode.BILINEAR)
            transforms.append(T.Resize((height, width), interpolation=interp, antialias=True))
        
        # Add normalization
        transforms.append(T.Normalize(
            mean=norm_config.get('mean', [0.485, 0.456, 0.406]),
            std=norm_config.get('std', [0.229, 0.224, 0.225])
        ))
        
        return T.Compose(transforms)
