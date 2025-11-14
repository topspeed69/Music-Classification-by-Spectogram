# CNN Module for Self-Supervised Music Learning

This module implements the deep learning pipeline for self-supervised music classification and recommendation.

## Structure

```
CNN/
├── models/              # CNN encoder and projection head
├── augmentation/        # Data augmentation techniques
├── data/               # Dataset and dataloaders
├── training/           # Training scripts and loss functions
├── embeddings/         # Embedding extraction
├── recommendation/     # Similarity search and recommendations
└── utils/              # Utility functions
```

## Quick Usage

### Training
```python
python training/train.py \
    --model_config ../configs/model_config.yaml \
    --training_config ../configs/training_config.yaml \
    --data_config ../configs/data_config.yaml
```

### Extract Embeddings
```python
python embeddings/extract_embeddings.py \
    --checkpoint ../checkpoints/best_model.pth \
    --data_dir ../AudioToSpectogram/output_mel \
    --output_dir ../embeddings_db
```

### Get Recommendations
```python
python recommendation/recommender.py \
    --query_song "path/to/spectrogram.png" \
    --model_checkpoint ../checkpoints/best_model.pth \
    --embeddings_db ../embeddings_db/embeddings.pkl \
    --top_k 10
```

## Key Components

### 1. Models (`models/`)
- **AudioEncoderCNN**: CNN encoder for spectrograms
- **ProjectionHead**: Maps features to embedding space
- **ContrastiveModel**: Complete model for training

### 2. Augmentation (`augmentation/`)
Implements 6 augmentation techniques:
- Random crop
- Color jitter
- Gaussian noise
- Horizontal flip (time reversal)
- Frequency masking
- Time masking

### 3. Training (`training/`)
- **NT-Xent Loss**: SimCLR contrastive loss
- **InfoNCE Loss**: Alternative contrastive loss
- **SupCon Loss**: Supervised contrastive loss

### 4. Recommendation (`recommendation/`)
- **SimilaritySearcher**: Cosine similarity search
- **FAISSSearcher**: Fast approximate search
- **MusicRecommender**: Complete recommendation system

See main [README.md](../README.md) for more details.
