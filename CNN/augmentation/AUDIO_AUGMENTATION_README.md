# Audio Augmentation Pipeline for Self-Supervised Contrastive Learning

This module implements a comprehensive two-stage augmentation pipeline optimized for music audio in self-supervised contrastive learning scenarios. The pipeline operates on mel spectrograms and applies realistic, musically-informed augmentations.

## Features

### Two-Stage Augmentation Architecture

#### Stage 1: Waveform Augmentations (Before Mel Transform)
Applied to raw audio waveforms before converting to spectrograms:

1. **Pitch Shift** - Shifts pitch by ±1-3 semitones
2. **Tempo Stretch** - Changes playback speed by ±5-12%
3. **Gain Adjustment** - Modifies volume by ±3-6 dB
4. **Parametric EQ Filtering** - Applies low-pass, high-pass, or band-pass filters
5. **Dynamic Range Compression** - Reduces dynamic range of audio signal
6. **Environmental Noise** - Adds realistic noise with SNR 10-30 dB
7. **Convolutional Reverb** - Applies room reverb effect

#### Stage 2: Spectrogram Augmentations (After Mel Transform)
Applied to mel spectrograms:

1. **Time Masking** (SpecAugment) - Masks random time steps
2. **Frequency Masking** (SpecAugment) - Masks random frequency bands
3. **Time Warping** - Warps time axis for temporal robustness

### Key Design Principles

- ✅ **Realistic augmentations** - No axis flips, rotations, or color jittering
- ✅ **Musically informed** - Augmentations preserve musical structure
- ✅ **Configurable probabilities** - Per-augmentation probability control
- ✅ **Random selection** - Randomly selects 3 waveform + 2 spectrogram augmentations per sample
- ✅ **Independent views** - Generates two independently augmented views for contrastive learning
- ✅ **Optimized for speed** - Worker pool limited to 7 threads
- ✅ **Robust error handling** - Gracefully handles augmentation failures

## Installation

Required dependencies:

```bash
pip install torch torchaudio librosa numpy scipy
```

## Quick Start

### Basic Usage

```python
from CNN.augmentation import ContrastiveAudioAugmentation

# Initialize augmentation pipeline
aug_pipeline = ContrastiveAudioAugmentation(
    sr=22050,                    # Sample rate
    n_mels=128,                  # Number of mel bands
    duration=3.0,                # Audio duration in seconds
    num_waveform_augs=3,         # Number of waveform augmentations to apply
    num_spectrogram_augs=2,      # Number of spectrogram augmentations to apply
    max_workers=7                # Maximum worker threads
)

# Process a single audio file
view1, view2 = aug_pipeline("path/to/audio.wav")
print(f"View 1 shape: {view1.shape}")  # [1, 128, time_steps]
print(f"View 2 shape: {view2.shape}")  # [1, 128, time_steps]
```

### Using with PyTorch Dataset

```python
from CNN.augmentation import AudioDatasetWithAugmentation
from torch.utils.data import DataLoader

# Create augmentation pipeline
aug_pipeline = ContrastiveAudioAugmentation(
    sr=22050,
    n_mels=128,
    duration=3.0,
    num_waveform_augs=3,
    num_spectrogram_augs=2,
    max_workers=7
)

# Create dataset (audio files are directly in fma_small_dataset directory)
dataset = AudioDatasetWithAugmentation(
    audio_dir="AudioToSpectogram/fma_small_dataset",
    augmentation=aug_pipeline,
    file_extensions=('.wav', '.mp3', '.flac', '.ogg')
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for views1, views2 in dataloader:
    # views1: [B, 1, n_mels, time]
    # views2: [B, 1, n_mels, time]
    loss = contrastive_loss(model(views1), model(views2))
    loss.backward()
```

### Custom Augmentation Probabilities

```python
# Define custom probabilities for waveform augmentations
waveform_probs = {
    'pitch_shift': 0.8,      # 80% chance to apply
    'tempo_stretch': 0.3,    # 30% chance to apply
    'gain': 0.9,
    'eq': 0.5,
    'compression': 0.2,
    'noise': 0.7,
    'reverb': 0.6
}

# Define custom probabilities for spectrogram augmentations
spectrogram_probs = {
    'time_mask': 0.9,
    'freq_mask': 0.9,
    'time_warp': 0.3
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
```

## Integration with Training

### Example: Contrastive Learning Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CNN.augmentation import ContrastiveAudioAugmentation, AudioDatasetWithAugmentation
from CNN.models import AudioEncoder
from CNN.training import ContrastiveLoss

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = ContrastiveLoss(temperature=0.07)

# Create augmentation pipeline
aug_pipeline = ContrastiveAudioAugmentation(
    sr=22050,
    n_mels=128,
    duration=3.0,
    num_waveform_augs=3,
    num_spectrogram_augs=2,
    max_workers=7
)

# Create dataset and dataloader
dataset = AudioDatasetWithAugmentation(
    audio_dir="AudioToSpectogram/fma_small_dataset",
    augmentation=aug_pipeline
)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch_idx, (views1, views2) in enumerate(dataloader):
        views1 = views1.to(device)
        views2 = views2.to(device)
        
        # Forward pass
        embeddings1 = model(views1)
        embeddings2 = model(views2)
        
        # Compute contrastive loss
        loss = criterion(embeddings1, embeddings2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

## Configuration Parameters

### ContrastiveAudioAugmentation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sr` | int | 22050 | Sample rate for audio |
| `n_fft` | int | 2048 | FFT window size |
| `hop_length` | int | 512 | Hop length for STFT |
| `n_mels` | int | 128 | Number of mel frequency bands |
| `duration` | float | 3.0 | Audio duration in seconds |
| `num_waveform_augs` | int | 3 | Number of waveform augmentations to apply |
| `num_spectrogram_augs` | int | 2 | Number of spectrogram augmentations to apply |
| `max_workers` | int | 7 | Maximum number of worker threads (capped at 7) |
| `waveform_aug_probs` | dict | See below | Probability for each waveform augmentation |
| `spectrogram_aug_probs` | dict | See below | Probability for each spectrogram augmentation |

### Default Probabilities

**Waveform Augmentations:**
```python
{
    'pitch_shift': 0.6,
    'tempo_stretch': 0.5,
    'gain': 0.7,
    'eq': 0.5,
    'compression': 0.4,
    'noise': 0.6,
    'reverb': 0.5
}
```

**Spectrogram Augmentations:**
```python
{
    'time_mask': 0.8,
    'freq_mask': 0.8,
    'time_warp': 0.5
}
```

## Performance Considerations

### Speed Optimization

1. **Worker Pool**: Limited to 7 workers to prevent thread contention
2. **Efficient Audio Loading**: Uses `librosa` with `kaiser_fast` resampling
3. **ThreadPoolExecutor**: Parallel batch processing for multiple files
4. **Error Handling**: Graceful failure handling to avoid blocking

### Memory Management

- Audio files are loaded on-demand
- Spectrograms are computed in-memory only
- No unnecessary caching or duplication
- Efficient tensor operations with PyTorch

### Recommended Settings

For training on GPU:
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,           # Adjust based on GPU memory
    shuffle=True,
    num_workers=4,           # 2-4 workers recommended
    pin_memory=True,         # Enable for faster GPU transfer
    drop_last=True,          # Drop incomplete batches
    persistent_workers=True  # Keep workers alive between epochs
)
```

## File Structure

```
CNN/augmentation/
├── __init__.py                  # Module exports
├── audio_augmentations.py       # Main augmentation pipeline
├── augmentations.py             # Original spectrogram augmentations
├── demo_augmentation.py         # Demo and examples
└── AUDIO_AUGMENTATION_README.md # This file
```

## Testing

Run the demo script to verify installation:

```bash
cd CNN/augmentation
python demo_augmentation.py
```

This will:
- Display all available augmentation types
- Show default and custom probability configurations
- Demonstrate single file and batch processing (if audio files are available)

## Troubleshooting

### Issue: Audio loading fails

**Solution**: Ensure `librosa` and `ffmpeg` are installed:
```bash
pip install librosa
# For Ubuntu/Debian
sudo apt-get install ffmpeg
# For macOS
brew install ffmpeg
```

### Issue: Out of memory errors

**Solution**: Reduce batch size or number of dataloader workers:
```python
dataloader = DataLoader(dataset, batch_size=16, num_workers=2)
```

### Issue: Augmentations too slow

**Solution**: Reduce number of augmentations or adjust probabilities:
```python
aug_pipeline = ContrastiveAudioAugmentation(
    num_waveform_augs=2,      # Reduce from 3
    num_spectrogram_augs=1    # Reduce from 2
)
```

## Best Practices

1. **Start with defaults**: The default probabilities are well-tuned for music
2. **Monitor augmentation time**: Profile your training loop to ensure augmentations aren't a bottleneck
3. **Use appropriate duration**: 3-5 seconds is typical for music classification
4. **Balance augmentation strength**: Too many augmentations can hurt performance
5. **Test on validation set**: Verify augmentations improve model generalization

## Citation

If you use this augmentation pipeline in your research, please cite:

```bibtex
@misc{contrastive_audio_augmentation,
  title={Two-Stage Augmentation Pipeline for Self-Supervised Music Learning},
  author={Your Name},
  year={2025}
}
```

## License

This code is part of the Music Classification by Spectrogram project.

## References

- [SpecAugment](https://arxiv.org/abs/1904.08779) - Time and frequency masking
- [SimCLR](https://arxiv.org/abs/2002.05709) - Contrastive learning framework
- [Librosa](https://librosa.org/) - Audio processing library
