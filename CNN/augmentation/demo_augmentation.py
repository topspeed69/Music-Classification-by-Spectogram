"""
Demo script for ContrastiveAudioAugmentation pipeline
Shows how to use the augmentation system for self-supervised learning
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from augmentation.audio_augmentations import (
    ContrastiveAudioAugmentation,
    AudioDatasetWithAugmentation
)


def visualize_augmentations(view1, view2, save_path='augmentation_comparison.png'):
    """
    Visualize two augmented views side by side
    
    Args:
        view1: First augmented view [1, n_mels, time]
        view2: Second augmented view [1, n_mels, time]
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Convert to numpy and remove channel dimension
    view1_np = view1.squeeze(0).numpy()
    view2_np = view2.squeeze(0).numpy()
    
    # Plot View 1
    im1 = axes[0].imshow(view1_np, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Augmented View 1', fontsize=14)
    axes[0].set_ylabel('Mel Frequency Bins')
    axes[0].set_xlabel('Time Frames')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot View 2
    im2 = axes[1].imshow(view2_np, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Augmented View 2', fontsize=14)
    axes[1].set_ylabel('Mel Frequency Bins')
    axes[1].set_xlabel('Time Frames')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def demo_single_audio():
    """
    Demo: Process a single audio file and visualize augmentations
    """
    print("="*60)
    print("DEMO 1: Single Audio File Processing")
    print("="*60)
    
    # Initialize augmentation pipeline
    aug_pipeline = ContrastiveAudioAugmentation(
        sr=22050,
        n_mels=128,
        duration=3.0,
        num_waveform_augs=3,      # Apply 3 random waveform augmentations
        num_spectrogram_augs=2,    # Apply 2 random spectrogram augmentations
        max_workers=7
    )
    
    # Path to audio file (update this path)
    # Note: fma_small_dataset has all audio files directly in the folder (no subfolders)
    audio_path = "AudioToSpectogram/fma_small_dataset/000002.mp3"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        print("Please update the audio_path variable with a valid audio file")
        print("Note: Audio files are directly in fma_small_dataset/ (flat structure)")
        return
    
    # Generate two augmented views
    print(f"\nProcessing: {audio_path}")
    view1, view2 = aug_pipeline(audio_path)
    
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    
    # Visualize
    visualize_augmentations(view1, view2, 'single_audio_augmentation.png')
    print("\n✓ Single audio processing completed!")


def demo_batch_processing():
    """
    Demo: Process multiple audio files in batch with DataLoader
    """
    print("\n" + "="*60)
    print("DEMO 2: Batch Processing with DataLoader")
    print("="*60)
    
    # Initialize augmentation pipeline
    aug_pipeline = ContrastiveAudioAugmentation(
        sr=22050,
        n_mels=128,
        duration=3.0,
        num_waveform_augs=3,
        num_spectrogram_augs=2,
        max_workers=7
    )
    
    # Path to audio directory (update this path)
    audio_dir = "AudioToSpectogram/fma_small_dataset"
    
    if not Path(audio_dir).exists():
        print(f"Audio directory not found: {audio_dir}")
        print("Please update the audio_dir variable with a valid directory")
        return
    
    # Create dataset
    try:
        dataset = AudioDatasetWithAugmentation(
            audio_dir=audio_dir,
            augmentation=aug_pipeline,
            file_extensions=('.wav', '.mp3', '.flac', '.ogg')
        )
        
        print(f"\nDataset size: {len(dataset)} audio files")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Process one batch
        print("\nProcessing one batch...")
        for batch_idx, (views1, views2) in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  Views 1 shape: {views1.shape}")  # [B, 1, n_mels, time]
            print(f"  Views 2 shape: {views2.shape}")  # [B, 1, n_mels, time]
            
            # Visualize first sample from batch
            visualize_augmentations(
                views1[0],
                views2[0],
                f'batch_sample_{batch_idx}.png'
            )
            
            if batch_idx >= 2:  # Process only 3 batches
                break
        
        print("\n✓ Batch processing completed!")
        
    except ValueError as e:
        print(f"Error: {e}")


def demo_custom_probabilities():
    """
    Demo: Use custom augmentation probabilities
    """
    print("\n" + "="*60)
    print("DEMO 3: Custom Augmentation Probabilities")
    print("="*60)
    
    # Custom probabilities for waveform augmentations
    waveform_probs = {
        'pitch_shift': 0.8,      # High probability
        'tempo_stretch': 0.3,    # Low probability
        'gain': 0.9,             # Very high probability
        'eq': 0.5,
        'compression': 0.2,      # Low probability
        'noise': 0.7,
        'reverb': 0.6
    }
    
    # Custom probabilities for spectrogram augmentations
    spectrogram_probs = {
        'time_mask': 0.9,        # Very high probability
        'freq_mask': 0.9,        # Very high probability
        'time_warp': 0.3         # Low probability
    }
    
    # Initialize with custom probabilities
    aug_pipeline = ContrastiveAudioAugmentation(
        sr=22050,
        n_mels=128,
        duration=3.0,
        num_waveform_augs=3,
        num_spectrogram_augs=2,
        max_workers=7,
        waveform_aug_probs=waveform_probs,
        spectrogram_aug_probs=spectrogram_probs
    )
    
    print("\nCustom augmentation pipeline initialized!")
    print("\nWaveform augmentation probabilities:")
    for name, prob in waveform_probs.items():
        print(f"  {name}: {prob:.1%}")
    
    print("\nSpectrogram augmentation probabilities:")
    for name, prob in spectrogram_probs.items():
        print(f"  {name}: {prob:.1%}")
    
    print("\n✓ Custom probabilities demo completed!")


def demo_augmentation_types():
    """
    Demo: Show what each augmentation does
    """
    print("\n" + "="*60)
    print("DEMO 4: Individual Augmentation Types")
    print("="*60)
    
    print("\nWaveform Augmentations (Stage 1 - Before Mel Transform):")
    print("  1. Pitch Shift: Shifts pitch by ±1-3 semitones")
    print("  2. Tempo Stretch: Changes playback speed by ±5-12%")
    print("  3. Gain Adjustment: Modifies volume by ±3-6 dB")
    print("  4. Parametric EQ: Applies low-pass, high-pass, or band-pass filtering")
    print("  5. Dynamic Range Compression: Reduces dynamic range of audio")
    print("  6. Add Noise: Adds environmental noise with SNR 10-30 dB")
    print("  7. Add Reverb: Applies convolutional reverb effect")
    
    print("\nSpectrogram Augmentations (Stage 2 - After Mel Transform):")
    print("  1. Time Masking: Masks random time steps (SpecAugment)")
    print("  2. Frequency Masking: Masks random frequency bands (SpecAugment)")
    print("  3. Time Warping: Warps time axis for temporal robustness")
    
    print("\nAugmentation Strategy:")
    print("  - Randomly select 3 waveform augmentations per sample")
    print("  - Randomly select 2 spectrogram augmentations per sample")
    print("  - Each augmentation applied with configurable probability")
    print("  - Two independent views generated for contrastive learning")
    print("  - Worker pool limited to 7 threads for efficiency")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("CONTRASTIVE AUDIO AUGMENTATION PIPELINE DEMO")
    print("="*60)
    
    # Show augmentation types
    demo_augmentation_types()
    
    # Demo with custom probabilities
    demo_custom_probabilities()
    
    # Uncomment these if you have audio files available:
    # demo_single_audio()
    # demo_batch_processing()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED!")
    print("="*60)
    print("\nTo use with your own audio:")
    print("1. Update audio paths in demo_single_audio() and demo_batch_processing()")
    print("2. Uncomment the demo function calls in main()")
    print("3. Run: python demo_augmentation.py")


if __name__ == "__main__":
    main()
