# Pre-Generation Quick Reference

## 🚀 Quick Setup

### 1. Enable in Notebook
```python
PRE_GENERATE = True
NUM_AUGMENTED_VERSIONS = 5
```

### 2. Run Generation Cell
- Takes 30-60 minutes (one-time)
- Saves to `data/augmented_spectrograms/`
- Progress bar shows ETA

### 3. Training Uses Pre-Generated Automatically
```python
# Automatically created by notebook
final_train_loader  # Uses pre-generated data
final_val_loader    # Uses pre-generated data
```

---

## 📊 Quick Comparison

| Feature | Pre-Generated | On-the-Fly |
|---------|--------------|------------|
| Training Speed | ⚡ **3-5x faster** | 1x baseline |
| Storage | 6-8 GB | Minimal |
| Setup Time | 30-60 min (once) | None |
| Variability | Fixed per version | Different each epoch |
| Reusability | ✅ Excellent | ❌ No |

---

## 💾 Storage Needs

| Files | Versions | Storage |
|-------|----------|---------|
| 8,000 | 3 | ~5 GB |
| 8,000 | 5 | ~7 GB |
| 8,000 | 10 | ~14 GB |

---

## 🔧 Common Commands

### Check if Files Exist
```python
from pathlib import Path
OUTPUT_DIR = Path('data/augmented_spectrograms')
if (OUTPUT_DIR / 'metadata.json').exists():
    print("✓ Pre-generated files found")
```

### Delete and Re-Generate
```bash
rm -rf data/augmented_spectrograms
# Then re-run generation cell
```

### Check Disk Space
```python
import shutil
total, used, free = shutil.disk_usage("/")
print(f"Free: {free / (1024**3):.1f} GB")
```

### Load a Single File
```python
import torch
data = torch.load('data/augmented_spectrograms/train/000002_v0.pt')
view1 = data['view1']  # [1, 128, 130]
view2 = data['view2']  # [1, 128, 130]
```

---

## 📁 File Structure

```
data/augmented_spectrograms/
├── train/
│   ├── 000002_v0.pt  # Audio 000002, version 0
│   ├── 000002_v1.pt  # Audio 000002, version 1
│   └── ...
├── val/
│   └── ...
└── metadata.json     # Config + file tracking
```

---

## ⚡ Performance

### Training Time (50 epochs)
- **On-the-Fly**: 12-16 hours
- **Pre-Generated**: **3-4 hours** (+ 1 hour setup)
- **Time Saved**: ~10 hours per full training run

---

## ⚠️ Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| Out of space | Reduce `NUM_AUGMENTED_VERSIONS` to 3 |
| Too slow | Check CPU usage, should use ~7 cores |
| Loading errors | Dataset has fallback, will skip bad files |
| Missing metadata | Re-run generation cell |

---

## ✅ When to Use Pre-Generation

**YES** ✅
- Multiple training runs
- Hyperparameter tuning
- Time-constrained (Colab)
- Have 8+ GB disk space

**NO** ❌
- Experimenting with augmentations
- Small datasets (< 1000 files)
- Limited disk space
- Want maximum variability

---

## 🎯 Recommended Settings

```python
# Balanced: Speed + Variety
PRE_GENERATE = True
NUM_AUGMENTED_VERSIONS = 5

# Fast Setup: Minimal storage
PRE_GENERATE = True
NUM_AUGMENTED_VERSIONS = 3

# Maximum Variety: More augmentations
PRE_GENERATE = True
NUM_AUGMENTED_VERSIONS = 10
```

---

## 📚 Full Documentation

See `PRE_GENERATION_GUIDE.md` for complete details, advanced usage, and troubleshooting.
