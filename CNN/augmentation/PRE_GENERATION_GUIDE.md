# Pre-Generation Guide for Augmented Spectrograms

## Overview

This guide explains how to pre-generate and reuse augmented spectrograms for faster training in your self-supervised contrastive learning pipeline.

## Table of Contents
- [Why Pre-Generate?](#why-pre-generate)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Usage in Notebook](#usage-in-notebook)
- [Storage Requirements](#storage-requirements)
- [Troubleshooting](#troubleshooting)

---

## Why Pre-Generate?

### Benefits ✅

1. **Faster Training**: No augmentation overhead during training
   - On-the-fly: ~1-2 seconds per batch (augmentation + forward pass)
   - Pre-generated: ~0.1-0.3 seconds per batch (just forward pass)
   - **Speed improvement: 5-10x faster**

2. **Reusability**: Generate once, use for multiple experiments
   - Train with different model architectures
   - Try different hyperparameters
   - Resume interrupted training

3. **Reproducibility**: Same augmentations across runs
   - Consistent results across experiments
   - Easier to compare model changes

4. **Resource Efficiency**: 
   - CPU-intensive augmentation done once
   - GPU can focus entirely on training
   - Better for cloud/Colab environments with limited compute time

### Trade-offs ⚠️

1. **Storage**: Requires 15-20 GB for 8000 audio files × 5 versions
2. **Fixed Augmentations**: Same augmentations each epoch (less variability)
3. **Initial Generation Time**: 30-60 minutes one-time cost

---

## Quick Start

### Step 1: Enable Pre-Generation in Notebook

In the notebook cell:

```python
# Configuration for pre-generating augmented spectrograms
PRE_GENERATE = True  # Set to True to enable pre-generation
NUM_AUGMENTED_VERSIONS = 5  # Number of augmented versions per audio file
OUTPUT_DIR = Path('data/augmented_spectrograms')
```

### Step 2: Run the Generation Cell

Execute the "Pre-generate augmented spectrograms" cell. This will:
- Create output directories (`data/augmented_spectrograms/train` and `val`)
- Generate `NUM_AUGMENTED_VERSIONS` versions of each audio file
- Save as PyTorch tensors (`.pt` files)
- Track metadata in `metadata.json`
- Show progress bar with ETA

**Example Output:**
```
============================================================
PRE-GENERATING AUGMENTED SPECTROGRAMS
============================================================

This will take some time but only needs to be done once!
You can use these pre-generated spectrograms for faster training.

============================================================
Generating TRAIN set
============================================================
Train: 100%|██████████| 32000/32000 [45:23<00:00, 11.74it/s]
✓ Train complete: 32000 files generated

============================================================
Generating VAL set
============================================================
Val: 100%|██████████| 8000/8000 [11:21<00:00, 11.74it/s]
✓ Val complete: 8000 files generated

============================================================
GENERATION COMPLETE!
============================================================
Total time: 56.7 minutes
Total files generated: 40000
Metadata saved to: data/augmented_spectrograms/metadata.json

You can now use these pre-generated spectrograms for fast training!
```

### Step 3: Use Pre-Generated Data

The notebook automatically detects pre-generated files and creates appropriate dataloaders:

```python
# The notebook will automatically use pre-generated files if available
final_train_loader  # Will use PreGeneratedSpectrogramDataset
final_val_loader    # Will use PreGeneratedSpectrogramDataset
```

---

## Configuration

### Key Parameters

```python
# Enable/disable pre-generation
PRE_GENERATE = True  # Set to False to use on-the-fly augmentation

# Number of augmented versions per audio file
NUM_AUGMENTED_VERSIONS = 5  # Recommended: 3-10
# - Lower (3): Less storage, less variety
# - Higher (10): More storage, more variety

# Output directory
OUTPUT_DIR = Path('data/augmented_spectrograms')
```

### Skipping Re-Generation

If files already exist, the notebook will detect them:

```python
# In configuration cell, it automatically sets:
SKIP_GENERATION = True  # If files already exist
```

To force re-generation, delete the output directory:

```bash
rm -rf data/augmented_spectrograms
```

---

## File Structure

### Generated Directory Layout

```
data/
└── augmented_spectrograms/
    ├── train/
    │   ├── 000002_v0.pt
    │   ├── 000002_v1.pt
    │   ├── 000002_v2.pt
    │   ├── 000002_v3.pt
    │   ├── 000002_v4.pt
    │   ├── 000005_v0.pt
    │   └── ...
    ├── val/
    │   └── [same structure]
    └── metadata.json
```

### File Naming Convention

Format: `{audio_name}_v{version}.pt`

- `audio_name`: Original audio file name (without extension)
- `version`: Augmentation version number (0 to NUM_AUGMENTED_VERSIONS-1)

Examples:
- `000002_v0.pt` → Version 0 of audio file `000002.mp3`
- `099999_v4.pt` → Version 4 of audio file `099999.mp3`

### Contents of Each `.pt` File

Each pre-generated file contains a dictionary:

```python
{
    'view1': torch.Tensor,      # First augmented mel spectrogram [1, 128, T]
    'view2': torch.Tensor,      # Second augmented mel spectrogram [1, 128, T]
    'audio_name': str,          # Original audio file name (e.g., '000002')
    'version': int,             # Augmentation version (0-4)
    'original_path': str        # Path to original audio file
}
```

**Loading Example:**
```python
import torch

# Load a pre-generated file
data = torch.load('data/augmented_spectrograms/train/000002_v0.pt')

view1 = data['view1']  # Shape: [1, 128, T]
view2 = data['view2']  # Shape: [1, 128, T]
```

### Metadata File (`metadata.json`)

Contains configuration and file tracking:

```json
{
  "train": [
    {
      "file": "data/augmented_spectrograms/train/000002_v0.pt",
      "audio_name": "000002",
      "version": 0,
      "shape": [1, 128, 130]
    },
    ...
  ],
  "val": [...],
  "config": {
    "sample_rate": 22050,
    "n_mels": 128,
    "duration": 3.0,
    "num_versions": 5,
    "num_waveform_augs": 3,
    "num_spectrogram_augs": 2
  }
}
```

---

## Usage in Notebook

### Complete Workflow

```python
# 1. Configure
PRE_GENERATE = True
NUM_AUGMENTED_VERSIONS = 5

# 2. Run generation (only needed once)
# Execute the generation cell - takes 30-60 minutes

# 3. Load pre-generated data
# The notebook automatically creates dataloaders
train_loader = final_train_loader  # Uses PreGeneratedSpectrogramDataset
val_loader = final_val_loader

# 4. Train your model
for epoch in range(num_epochs):
    for view1, view2 in train_loader:
        # view1, view2 are already augmented mel spectrograms
        # No augmentation overhead - much faster!
        ...
```

### Switching Between Pre-Generated and On-the-Fly

```python
# Option 1: Use pre-generated (FAST)
USE_PREGENERATED = True
if USE_PREGENERATED and (OUTPUT_DIR / 'metadata.json').exists():
    train_loader, val_loader = create_pregenerated_dataloaders(...)
    
# Option 2: Use on-the-fly (FLEXIBLE)
USE_PREGENERATED = False
if not USE_PREGENERATED:
    # Use original dataloaders with augmentation
    train_loader = train_loader  # On-the-fly augmentation
    val_loader = val_loader
```

### Custom Dataset Class

The notebook includes `PreGeneratedSpectrogramDataset`:

```python
class PreGeneratedSpectrogramDataset(Dataset):
    """Dataset that loads pre-generated augmented spectrograms"""
    
    def __init__(self, metadata_list):
        self.files = [item['file'] for item in metadata_list]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data['view1'], data['view2']
```

---

## Storage Requirements

### Calculation

Each audio file generates:
- `NUM_AUGMENTED_VERSIONS` versions
- Each version has 2 views (view1, view2)
- Each view is a mel spectrogram: `[1, 128, T]` where T ≈ 130 frames

**Size per view:**
- Shape: `[1, 128, 130]`
- Data type: `float32` (4 bytes)
- Size: 1 × 128 × 130 × 4 = 65,536 bytes ≈ 64 KB

**Size per version:**
- 2 views + metadata ≈ 128 KB + 1 KB = 129 KB

**Total size:**
- 8000 audio files × 5 versions × 129 KB = **5.16 GB** (theoretical minimum)
- With PyTorch overhead: **~6-8 GB** (actual)

### For Different Configurations

| Audio Files | Versions | Train/Val Split | Total Storage |
|------------|----------|-----------------|---------------|
| 8,000 | 3 | 80/20 | ~4-5 GB |
| 8,000 | 5 | 80/20 | ~6-8 GB |
| 8,000 | 10 | 80/20 | ~12-16 GB |
| 16,000 | 5 | 80/20 | ~12-16 GB |

### Disk Space Check

The notebook automatically checks available space:

```python
# Check available disk space
import shutil
total, used, free = shutil.disk_usage("/")
print(f"Available space: {free / (1024**3):.1f} GB")
```

---

## Troubleshooting

### Issue 1: Out of Disk Space

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**
1. Reduce `NUM_AUGMENTED_VERSIONS` (try 3 instead of 5)
2. Free up disk space
3. Change `OUTPUT_DIR` to a different drive with more space
4. Use on-the-fly augmentation instead (`PRE_GENERATE = False`)

### Issue 2: Generation Takes Too Long

**Expected time:** 30-60 minutes for 8000 files

**If it's taking longer:**
1. Check CPU usage - should be using multiple cores
2. Check if `max_workers=7` is set correctly
3. Verify audio files are not corrupted (check logs)
4. Consider reducing `NUM_AUGMENTED_VERSIONS`

### Issue 3: Errors During Generation

**The generation cell includes error handling:**
- Skips problematic audio files
- Continues with remaining files
- Shows error count in progress bar

**To investigate errors:**
```python
# Check the progress bar output
# Errors: X  <- Number of files that failed

# Common causes:
# - Corrupted audio files
# - Unsupported audio format
# - Memory issues
```

### Issue 4: Different Augmentations Than Expected

**Remember:** Pre-generated augmentations are FIXED per version.

If you want different augmentations:
1. Delete existing files: `rm -rf data/augmented_spectrograms`
2. Modify augmentation config in `configs/audio_augmentation_config.yaml`
3. Re-run generation

### Issue 5: Loading Errors During Training

**Symptoms:**
```
Error loading data/augmented_spectrograms/train/000002_v0.pt
```

**Solutions:**
1. Check if file exists and is not corrupted
2. Verify PyTorch version compatibility
3. Re-generate that specific file
4. The dataset includes fallback logic - it will load a random other file

### Issue 6: Metadata File Missing

**Symptoms:**
```
⚠ No metadata file found
```

**Solutions:**
1. Re-run generation cell
2. Check `OUTPUT_DIR` path is correct
3. Verify write permissions

---

## Advanced Usage

### Generating Only Specific Splits

```python
# Generate only training set
def generate_only_train():
    metadata['train'] = generate_and_save(train_indices, TRAIN_OUTPUT_DIR, 'train')
    with open(OUTPUT_DIR / 'metadata_train.json', 'w') as f:
        json.dump(metadata, f, indent=2)

# Generate only validation set
def generate_only_val():
    metadata['val'] = generate_and_save(val_indices, VAL_OUTPUT_DIR, 'val')
    with open(OUTPUT_DIR / 'metadata_val.json', 'w') as f:
        json.dump(metadata, f, indent=2)
```

### Parallel Generation

For faster generation, you can split the work:

```python
# Split indices into chunks
chunk_size = len(train_indices) // 4
chunks = [train_indices[i:i+chunk_size] for i in range(0, len(train_indices), chunk_size)]

# Generate each chunk (can be done on different machines)
for i, chunk in enumerate(chunks):
    generate_and_save(chunk, TRAIN_OUTPUT_DIR, f'train_chunk_{i}')
```

### Custom Augmentation Per Version

If you want different augmentation strategies per version:

```python
# Modify the generation loop
for version in range(NUM_AUGMENTED_VERSIONS):
    # Adjust augmentation config based on version
    if version < 2:
        aug_pipeline.num_waveform_augs = 2  # Lighter augmentation
    else:
        aug_pipeline.num_waveform_augs = 4  # Heavier augmentation
    
    view1, view2 = full_dataset[idx]
    # ... save ...
```

---

## Performance Comparison

### Training Speed (per epoch)

| Method | Time per Epoch | Speed |
|--------|---------------|-------|
| On-the-Fly Augmentation | ~15-20 minutes | 1x |
| Pre-Generated | ~3-5 minutes | **3-5x faster** |

### Total Training Time (50 epochs)

| Method | Total Time | Notes |
|--------|-----------|-------|
| On-the-Fly | 12-16 hours | Flexible, variable augmentations |
| Pre-Generated | **2.5-4 hours + 1 hour setup** | Fixed augmentations |

**Recommendation:** 
- For **single experiments**: On-the-fly might be sufficient
- For **multiple experiments** or **hyperparameter tuning**: Pre-generated saves significant time

---

## Best Practices

1. **Start with 5 versions**: Good balance of variety and storage
2. **Generate once, experiment many times**: Reuse for multiple model architectures
3. **Monitor disk space**: Ensure you have 8-10 GB free before starting
4. **Keep metadata.json**: Essential for loading pre-generated data
5. **Document augmentation config**: Save configs with your results
6. **Validate first**: Run with on-the-fly on small subset to verify pipeline
7. **Backup metadata**: Copy `metadata.json` to safe location

---

## Summary

Pre-generation is a powerful technique for speeding up training in self-supervised contrastive learning:

✅ **Use pre-generation when:**
- Running multiple experiments with same augmentation config
- Training large models (many epochs)
- Working in time-constrained environments (Colab)
- You have sufficient disk space (8+ GB)

❌ **Use on-the-fly when:**
- Exploring different augmentation strategies
- Disk space is limited
- You want maximum variability (different augs each epoch)
- Dataset is small (< 1000 files)

The notebook supports both approaches seamlessly - choose what works best for your workflow!
