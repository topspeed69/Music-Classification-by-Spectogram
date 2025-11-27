# Complete Usage Example - Audio Augmentation with fma_small_dataset

This guide shows how to use the audio augmentation pipeline with the actual `fma_small_dataset` structure.

## Dataset Structure

The `fma_small_dataset` has **all 8000 audio files directly in one folder**:

```
AudioToSpectogram/
  fma_small_dataset/
    000002.mp3
    000005.mp3
    000010.mp3
    000140.mp3
    ... (8000 files total)
```

## Quick Test

### 1. Test Your Setup

```bash
cd CNN/augmentation
python test_augmentation.py
```

This will verify:
- ✅ Dataset exists and has audio files
- ✅ Augmentation modules import correctly
- ✅ Pipeline initializes
- ✅ Can process audio files
- ✅ Can create PyTorch datasets
- ✅ Train/val split works

### 2. Run the Demo

```bash
python demo_augmentation.py
```

This shows:
- Available augmentation types
- Custom probability configuration
- Example outputs

## Training Examples

### Example 1: Simple Training with Auto-Split

Train on all 8000 files with automatic 80/20 split:

```bash
python train_with_audio_aug.py \
    --train-dir ../../AudioToSpectogram/fma_small_dataset \
    --split 0.8 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4
```

Result:
- Training: 6400 samples (80%)
- Validation: 1600 samples (20%)

### Example 2: Custom Split Ratio

Train with 90/10 split:

```bash
python train_with_audio_aug.py \
    --train-dir ../../AudioToSpectogram/fma_small_dataset \
    --split 0.9 \
    --epochs 50 \
    --batch-size 64
```

### Example 3: Using Configuration File

```bash
python train_with_audio_aug.py \
    --train-dir ../../AudioToSpectogram/fma_small_dataset \
    --config ../../configs/audio_augmentation_config.yaml \
    --split 0.8 \
    --epochs 100
```

### Example 4: GPU Training with Larger Batch

```bash
python train_with_audio_aug.py \
    --train-dir ../../AudioToSpectogram/fma_small_dataset \
    --split 0.8 \
    --epochs 100 \
    --batch-size 64 \
    --device cuda
```

## Code Examples

### Example 1: Basic Usage

```python
from CNN.augmentation import ContrastiveAudioAugmentation

# Initialize
aug = ContrastiveAudioAugmentation(
    sr=22050,
    n_mels=128,
    duration=3.0,
    num_waveform_augs=3,
    num_spectrogram_augs=2,
    max_workers=7
)

# Process one file
view1, view2 = aug("AudioToSpectogram/fma_small_dataset/000002.mp3")
print(f"View 1: {view1.shape}")  # [1, 128, time]
print(f"View 2: {view2.shape}")  # [1, 128, time]
```

### Example 2: Create Dataset with Split

```python
from CNN.augmentation import AudioDatasetWithAugmentation, ContrastiveAudioAugmentation
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Create augmentation
aug = ContrastiveAudioAugmentation(
    sr=22050,
    n_mels=128,
    duration=3.0,
    num_waveform_augs=3,
    num_spectrogram_augs=2
)

# Load full dataset (all 8000 files)
full_dataset = AudioDatasetWithAugmentation(
    audio_dir="AudioToSpectogram/fma_small_dataset",
    augmentation=aug
)

print(f"Total samples: {len(full_dataset)}")  # 8000

# Split 80/20
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(
    indices, train_size=0.8, random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")  # 6400, 1600

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

### Example 3: Training Loop

```python
import torch
from CNN.models import AudioEncoder
from CNN.training import ContrastiveLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = AudioEncoder(input_channels=1, embedding_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = ContrastiveLoss(temperature=0.07)

# Training loop
for epoch in range(100):
    model.train()
    train_loss = 0
    
    for batch_idx, (views1, views2) in enumerate(train_loader):
        # Move to device
        views1 = views1.to(device)  # [B, 1, 128, time]
        views2 = views2.to(device)  # [B, 1, 128, time]
        
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
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for views1, views2 in val_loader:
            views1, views2 = views1.to(device), views2.to(device)
            embeddings1 = model(views1)
            embeddings2 = model(views2)
            loss = criterion(embeddings1, embeddings2)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
          f"Val Loss = {val_loss/len(val_loader):.4f}")
```

### Example 4: Custom Augmentation Probabilities

```python
# High augmentation strength
aug_strong = ContrastiveAudioAugmentation(
    sr=22050,
    n_mels=128,
    num_waveform_augs=4,  # More augmentations
    num_spectrogram_augs=3,
    waveform_aug_probs={
        'pitch_shift': 0.9,
        'tempo_stretch': 0.8,
        'gain': 1.0,
        'eq': 0.7,
        'compression': 0.6,
        'noise': 0.8,
        'reverb': 0.7
    },
    spectrogram_aug_probs={
        'time_mask': 1.0,
        'freq_mask': 1.0,
        'time_warp': 0.8
    }
)

# Low augmentation strength
aug_weak = ContrastiveAudioAugmentation(
    sr=22050,
    n_mels=128,
    num_waveform_augs=2,  # Fewer augmentations
    num_spectrogram_augs=1,
    waveform_aug_probs={
        'pitch_shift': 0.3,
        'tempo_stretch': 0.2,
        'gain': 0.5,
        'eq': 0.2,
        'compression': 0.1,
        'noise': 0.3,
        'reverb': 0.2
    },
    spectrogram_aug_probs={
        'time_mask': 0.5,
        'freq_mask': 0.5,
        'time_warp': 0.2
    }
)
```

## Performance Tuning

### For CPU Training

```python
# Smaller batch, fewer workers
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=2,
    pin_memory=False
)

# Reduce augmentation workers
aug = ContrastiveAudioAugmentation(
    max_workers=4  # Reduce from 7
)
```

### For GPU Training

```python
# Larger batch, more workers
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# Full augmentation workers
aug = ContrastiveAudioAugmentation(
    max_workers=7
)
```

## Expected Training Time

With the full fma_small_dataset (8000 files):

| Setup | Batch Size | Epoch Time | GPU Memory |
|-------|-----------|-----------|------------|
| CPU (8 cores) | 16 | ~45 min | N/A |
| CPU (8 cores) | 32 | ~40 min | N/A |
| GPU (RTX 3080) | 32 | ~8 min | ~4 GB |
| GPU (RTX 3080) | 64 | ~5 min | ~6 GB |

## Troubleshooting

### Issue: "No audio files found"

```bash
# Check if files exist
ls AudioToSpectogram/fma_small_dataset/*.mp3 | wc -l
# Should show 8000
```

### Issue: "Out of memory"

```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=16)

# Or reduce workers
aug = ContrastiveAudioAugmentation(max_workers=4)
```

### Issue: Slow augmentation

```python
# Reduce number of augmentations
aug = ContrastiveAudioAugmentation(
    num_waveform_augs=2,      # Instead of 3
    num_spectrogram_augs=1    # Instead of 2
)
```

## Next Steps

1. ✅ **Test your setup**: Run `python test_augmentation.py`
2. ✅ **Try the demo**: Run `python demo_augmentation.py`
3. 🚀 **Start training**: Run the training script with auto-split
4. 📊 **Monitor progress**: Check tensorboard logs in `runs/`
5. 💾 **Use checkpoints**: Load best model from `checkpoints/`

## Full Training Command

Ready to train? Run this:

```bash
cd CNN/augmentation

# Test first
python test_augmentation.py

# If tests pass, start training
python train_with_audio_aug.py \
    --train-dir ../../AudioToSpectogram/fma_small_dataset \
    --split 0.8 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir ../../checkpoints
```

Training will:
- Load 8000 audio files
- Split into 6400 train / 1600 val
- Apply augmentations on-the-fly
- Save checkpoints every 10 epochs
- Save best model based on validation loss

Happy training! 🎵🎶
