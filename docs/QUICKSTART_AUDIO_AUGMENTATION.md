# Quick Start Guide - Audio Augmentation Pipeline

> **Note:** All commands in this guide should be executed from the project root directory.

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Run the setup script
cd Music-Classification-by-Spectogram
./setup_audio_augmentation.sh

# OR manually install
pip install torch librosa soundfile numpy scipy matplotlib pyyaml
```

### Step 2: Prepare Your Audio Data

The `fma_small_dataset` has all audio files in a flat directory structure:

```
AudioToSpectogram/
  fma_small_dataset/
    000002.mp3
    000005.mp3
    000010.mp3
    ... (all 8000 files directly here)
```

**Note**: All audio files are directly in `fma_small_dataset/` with no subfolders.

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

**For train/val split**, see the split example below.

### How to Split the Dataset

Since all audio files are in one folder, use the automatic split feature:

```python
from CNN.augmentation import AudioDatasetWithAugmentation, ContrastiveAudioAugmentation
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Create augmentation
aug = ContrastiveAudioAugmentation()

# Load full dataset
full_dataset = AudioDatasetWithAugmentation(
    audio_dir="AudioToSpectogram/fma_small_dataset",
    augmentation=aug
)

# Split 80% train, 20% val
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

Or use the built-in split in the training script (see Option C above).

### Step 3: Run a Demo

```bash
cd CNN/augmentation
python demo_augmentation.py
```

This will show you all available augmentations and their usage.

### Step 4: Use in Your Code

#### Option A: Simple Usage

```python
from CNN.augmentation import ContrastiveAudioAugmentation

# Initialize
aug = ContrastiveAudioAugmentation(
    sr=22050,
    n_mels=128,
    duration=3.0,
    num_waveform_augs=3,
    num_spectrogram_augs=2
)

# Process audio
view1, view2 = aug("path/to/audio.mp3")
```

#### Option B: With PyTorch Dataset

```python
from CNN.augmentation import AudioDatasetWithAugmentation
from torch.utils.data import DataLoader

# Create dataset
dataset = AudioDatasetWithAugmentation(
    audio_dir="AudioToSpectogram/fma_small_dataset",
    augmentation=aug
)

# Create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
for views1, views2 in loader:
    # Your training code here
    pass
```

#### Option C: Complete Training Pipeline

**With automatic 80/20 train/val split:**
```bash
cd CNN/augmentation
python train_with_audio_aug.py \
    --train-dir ../../AudioToSpectogram/fma_small_dataset \
    --split 0.8 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4
```

**With separate train/val directories (if you've manually split):**
```bash
python train_with_audio_aug.py \
    --train-dir ../../AudioToSpectogram/fma_small_dataset/train \
    --val-dir ../../AudioToSpectogram/fma_small_dataset/val \
    --epochs 100 \
    --batch-size 32
```

## 📚 Documentation

- **Full Documentation**: [CNN/augmentation/AUDIO_AUGMENTATION_README.md](CNN/augmentation/AUDIO_AUGMENTATION_README.md)
- **Implementation Summary**: [AUDIO_AUGMENTATION_SUMMARY.md](AUDIO_AUGMENTATION_SUMMARY.md)
- **Configuration Example**: [configs/audio_augmentation_config.yaml](configs/audio_augmentation_config.yaml)

## 🎵 What Augmentations Are Applied?

### Stage 1: Waveform (3 randomly selected)
- Pitch shift (±1-3 semitones)
- Tempo stretch (±5-12%)
- Gain adjustment (±3-6 dB)
- Parametric EQ filtering
- Dynamic range compression
- Environmental noise (SNR 10-30 dB)
- Convolutional reverb

### Stage 2: Spectrogram (2 randomly selected)
- Time masking (SpecAugment)
- Frequency masking (SpecAugment)
- Time warping

## ⚙️ Configuration

Customize augmentation probabilities:

```python
aug = ContrastiveAudioAugmentation(
    sr=22050,
    n_mels=128,
    num_waveform_augs=3,
    num_spectrogram_augs=2,
    waveform_aug_probs={
        'pitch_shift': 0.8,    # 80% chance
        'tempo_stretch': 0.5,  # 50% chance
        'gain': 0.9,
        'eq': 0.5,
        'compression': 0.4,
        'noise': 0.7,
        'reverb': 0.6
    },
    spectrogram_aug_probs={
        'time_mask': 0.9,
        'freq_mask': 0.9,
        'time_warp': 0.5
    }
)
```

Or use a YAML config file:

```python
import yaml

with open('configs/audio_augmentation_config.yaml') as f:
    config = yaml.safe_load(f)

# Use config values...
```

## 🔍 Verify Installation

```python
# Test import
from CNN.augmentation import ContrastiveAudioAugmentation

# Test initialization
aug = ContrastiveAudioAugmentation()
print("✓ Installation successful!")

# Test with your audio
view1, view2 = aug("path/to/test.mp3")
print(f"✓ Generated views: {view1.shape}, {view2.shape}")
```

## 🐛 Troubleshooting

### Import errors
```bash
pip install torch librosa soundfile numpy scipy
```

### Audio loading fails
```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### Out of memory
```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=16, num_workers=2)

# Reduce workers
aug = ContrastiveAudioAugmentation(max_workers=4)
```

## 📊 Performance Tips

### For CPU Training
```python
dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=False
)
```

### For GPU Training
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## 💡 Key Features

✅ **Realistic augmentations** - No unrealistic transformations  
✅ **Musically informed** - Preserves musical structure  
✅ **Configurable** - Per-augmentation probability control  
✅ **Fast** - Optimized with worker pool (max 7 threads)  
✅ **Easy to use** - Drop-in PyTorch Dataset  
✅ **Well documented** - Comprehensive docs and examples  

## 📝 Examples

### Example 1: Basic Usage
```python
from CNN.augmentation import ContrastiveAudioAugmentation

aug = ContrastiveAudioAugmentation()
view1, view2 = aug("audio.mp3")
```

### Example 2: Custom Probabilities
```python
aug = ContrastiveAudioAugmentation(
    waveform_aug_probs={'pitch_shift': 1.0, 'gain': 0.8},
    spectrogram_aug_probs={'time_mask': 0.9, 'freq_mask': 0.9}
)
```

### Example 3: Training Loop
```python
from torch.utils.data import DataLoader
from CNN.augmentation import AudioDatasetWithAugmentation

dataset = AudioDatasetWithAugmentation("audio_dir", aug)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(100):
    for views1, views2 in loader:
        embeddings1 = model(views1.to(device))
        embeddings2 = model(views2.to(device))
        loss = criterion(embeddings1, embeddings2)
        loss.backward()
        optimizer.step()
```

## 🎯 What's Next?

1. ✅ Setup complete - Install dependencies
2. ✅ Data ready - Organize audio files
3. ✅ Test working - Run demo
4. 🚀 **Start training** - Use the training script
5. 📈 **Monitor results** - Check training curves
6. 🎨 **Fine-tune** - Adjust augmentation probabilities

## 📞 Need Help?

- Read the full docs: [AUDIO_AUGMENTATION_README.md](CNN/augmentation/AUDIO_AUGMENTATION_README.md)
- Check examples: [demo_augmentation.py](CNN/augmentation/demo_augmentation.py)
- Review summary: [AUDIO_AUGMENTATION_SUMMARY.md](AUDIO_AUGMENTATION_SUMMARY.md)

---

**Ready to train?** Run `python CNN/augmentation/train_with_audio_aug.py --help`
