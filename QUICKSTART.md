# Quick Start Guide

This guide will help you get started with the Music Classification and Recommendation System.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg installed

## Installation

1. **Install dependencies:**
```powershell
pip install -r requirements.txt
cd CNN
pip install -r requirements.txt
cd ..
```

## Step-by-Step Workflow

### Step 1: Convert Audio to Spectrograms

```powershell
cd AudioToSpectogram
python audio_to_spectogram_mel.py --source fma_small_dataset --dest output_mel --duration 30 --n_mels 128
cd ..
```

### Step 2: Train the Model

```powershell
cd CNN
python training/train.py --model_config ../configs/model_config.yaml --training_config ../configs/training_config.yaml --data_config ../configs/data_config.yaml
cd ..
```

Monitor training with TensorBoard:
```powershell
tensorboard --logdir logs/tensorboard
```

### Step 3: Extract Embeddings

```powershell
cd CNN
python embeddings/extract_embeddings.py --checkpoint ../checkpoints/best_model.pth --data_dir ../AudioToSpectogram/output_mel --output_dir ../embeddings_db --batch_size 32
cd ..
```

### Step 4: Get Recommendations

```powershell
cd CNN
python recommendation/recommender.py --query_song "path/to/spectrogram.png" --model_checkpoint ../checkpoints/best_model.pth --embeddings_db ../embeddings_db/embeddings.pkl --top_k 10
cd ..
```

## Python API Usage

### Extract Embeddings
```python
from CNN.models import build_model
from CNN.embeddings import extract_embeddings

# Load model and extract embeddings
model = build_model(config)
embeddings, paths = extract_embeddings(model, dataloader, device)
```

### Get Recommendations
```python
from CNN.recommendation import MusicRecommender

# Initialize recommender
recommender = MusicRecommender(
    model_path='checkpoints/best_model.pth',
    embeddings_db='embeddings_db/embeddings.pkl'
)

# Get recommendations
recommendations = recommender.recommend_from_spectrogram(
    spectrogram_path='path/to/song_spectrogram.png',
    top_k=10
)

for rank, (path, similarity) in enumerate(recommendations, 1):
    print(f"{rank}. {path} (similarity: {similarity:.4f})")
```

## Configuration

Edit configuration files in `configs/`:
- `model_config.yaml` - Model architecture
- `training_config.yaml` - Training hyperparameters
- `data_config.yaml` - Data and augmentation settings

## Troubleshooting

### Out of Memory
- Reduce batch size in `configs/training_config.yaml`
- Reduce image resolution in data config

### Slow Training
- Increase `num_workers` in data config
- Enable mixed precision training
- Use smaller model (reduce conv layers)

### Poor Recommendations
- Train for more epochs
- Adjust augmentation strength
- Check embedding normalization
- Increase embedding dimension

## Next Steps

1. Experiment with different augmentations
2. Try different CNN architectures
3. Implement FAISS for faster search
4. Add genre classification head
5. Create web interface

For detailed documentation, see the main [README.md](README.md).
