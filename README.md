# Music Classification and Recommendation System
## Self-Supervised Learning on Audio Spectrograms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project that uses self-supervised contrastive learning on audio spectrograms to classify music and recommend similar songs based on learned embeddings.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Audio to Spectrogram Conversion](#1-audio-to-spectrogram-conversion)
  - [2. Data Augmentation](#2-data-augmentation)
  - [3. Model Training](#3-model-training)
  - [4. Embedding Extraction](#4-embedding-extraction)
  - [5. Song Recommendation](#5-song-recommendation)
- [Technical Details](#technical-details)
- [Dataset](#dataset)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **self-supervised learning** approach to learn meaningful audio representations without requiring labeled data. By training a CNN encoder with contrastive learning on augmented spectrograms, the model learns to identify similar songs and can be used for:

- **Music Classification**: Categorize songs by genre, mood, or style
- **Song Recommendation**: Find similar songs based on audio features
- **Audio Similarity Search**: Retrieve songs that sound alike

The system leverages the power of **contrastive learning** to create robust audio embeddings that capture the essence of musical content.

---

## 🏗️ Pipeline Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Audio     │─────▶│ Spectrogram  │─────▶│  Augmentation   │
│   Files     │      │  Conversion  │      │   (5 types)     │
└─────────────┘      └──────────────┘      └─────────────────┘
                                                     │
                                                     ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Cosine    │◀─────│  Embeddings  │◀─────│   CNN Encoder   │
│ Similarity  │      │  Extraction  │      │                 │
└─────────────┘      └──────────────┘      └─────────────────┘
       │                                             ▲
       ▼                                             │
┌─────────────┐                            ┌─────────────────┐
│    Song     │                            │  Contrastive    │
│Recommendation│                            │     Loss        │
└─────────────┘                            │  (SimCLR/MoCo)  │
                                           └─────────────────┘
```

### Pipeline Stages:

1. **Audio → Spectrogram**: Convert audio files to mel-spectrograms
2. **Augmentation**: Apply 3 random augmentations per sample
3. **CNN Encoder**: Extract high-level audio features
4. **Contrastive Learning**: Train with self-supervised contrastive loss
5. **Embeddings**: Generate fixed-size vector representations
6. **Similarity Search**: Find similar songs using cosine similarity

---

## ✨ Key Features

### Self-Supervised Learning
- **No labels required**: Learn from raw audio data
- **Contrastive learning**: SimCLR-based approach
- **Data efficiency**: Learn robust representations with limited data

### Advanced Data Augmentation
The system applies 3 random augmentations to each spectrogram:
1. **Random Crop**: Extract sub-regions to learn position invariance
2. **Color Jitter**: Simulate intensity variations in frequency content
3. **Gaussian Noise**: Add random noise for robustness
4. **Horizontal Flip**: Time reversal to learn temporal invariance
5. **Frequency/Time Masking**: SpecAugment-style masking for regularization

### Robust CNN Encoder
- Deep convolutional architecture optimized for spectrograms
- Learns hierarchical audio features
- Produces compact, discriminative embeddings

### Efficient Recommendation System
- Fast cosine similarity search
- Scalable to large music databases
- Real-time song recommendations

---

## 📁 Project Structure

```
Music Classification by spectogram/
│
├── README.md                          # Main project documentation (this file)
│
├── AudioToSpectogram/                 # Spectrogram conversion module
│   ├── audio_to_spectogram.py         # Basic spectrogram converter
│   ├── audio_to_spectogram_mel.py     # Mel-spectrogram converter
│   ├── requirements.txt               # Dependencies for audio processing
│   ├── README.md                      # Audio conversion documentation
│   ├── assets/                        # Sample images/videos
│   ├── fma_small_dataset/             # Audio dataset
│   ├── output/                        # Regular spectrograms
│   └── output_mel/                    # Mel-spectrograms
│
├── CNN/                               # Deep learning pipeline
│   ├── models/                        # CNN encoder architectures
│   │   ├── __init__.py
│   │   ├── encoder.py                 # CNN encoder definition
│   │   └── projection_head.py         # Contrastive learning head
│   │
│   ├── augmentation/                  # Data augmentation module
│   │   ├── __init__.py
│   │   ├── augmentations.py           # Augmentation implementations
│   │   └── spec_augment.py            # SpecAugment implementation
│   │
│   ├── data/                          # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py                 # PyTorch dataset class
│   │   └── dataloader.py              # Data loader utilities
│   │
│   ├── training/                      # Training scripts
│   │   ├── __init__.py
│   │   ├── train.py                   # Main training script
│   │   ├── contrastive_loss.py        # Loss functions (NT-Xent)
│   │   └── config.py                  # Training configuration
│   │
│   ├── embeddings/                    # Embedding extraction
│   │   ├── __init__.py
│   │   ├── extract_embeddings.py      # Extract embeddings from audio
│   │   └── embedding_store.py         # Store and manage embeddings
│   │
│   ├── recommendation/                # Recommendation system
│   │   ├── __init__.py
│   │   ├── similarity_search.py       # Cosine similarity search
│   │   └── recommender.py             # Song recommendation engine
│   │
│   ├── utils/                         # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── visualization.py           # Plotting and visualization
│   │
│   └── requirements.txt               # Deep learning dependencies
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Dataset analysis
│   ├── 02_augmentation_demo.ipynb     # Visualize augmentations
│   ├── 03_training_experiments.ipynb  # Training experiments
│   └── 04_recommendation_demo.ipynb   # Demo recommendation system
│
├── configs/                           # Configuration files
│   ├── model_config.yaml              # Model hyperparameters
│   ├── training_config.yaml           # Training settings
│   └── data_config.yaml               # Data processing config
│
├── checkpoints/                       # Model checkpoints
│   └── .gitkeep
│
├── embeddings_db/                     # Stored embeddings
│   └── .gitkeep
│
└── requirements.txt                   # Global dependencies
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- FFmpeg (for audio processing)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Music Classification by spectogram"
```

2. **Create a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
# Install global requirements
pip install -r requirements.txt

# Install audio processing dependencies
cd AudioToSpectogram
pip install -r requirements.txt
cd ..

# Install deep learning dependencies
cd CNN
pip install -r requirements.txt
cd ..
```

4. **Install FFmpeg** (if not already installed)
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

---

## 📖 Usage

### 1. Audio to Spectrogram Conversion

Convert your audio files to mel-spectrograms:

```bash
cd AudioToSpectogram

# Convert audio files to mel-spectrograms
python audio_to_spectogram_mel.py \
    --source fma_small_dataset \
    --dest output_mel \
    --duration 30 \
    --n_mels 128
```

**Arguments:**
- `--source`: Directory containing audio files
- `--dest`: Output directory for spectrograms
- `--duration`: Audio duration in seconds (default: 30)
- `--n_mels`: Number of mel frequency bins (default: 128)

### 2. Data Augmentation

The augmentation pipeline automatically applies during training. Each spectrogram receives 3 random augmentations from:

- **Random Crop**: Crops a random region from the spectrogram
- **Color Jitter**: Adjusts brightness, contrast, and saturation
- **Gaussian Noise**: Adds Gaussian noise to simulate real-world conditions
- **Horizontal Flip**: Reverses the time axis
- **Frequency/Time Masking**: Masks random frequency bands or time steps (SpecAugment)

### 3. Model Training

Train the CNN encoder with contrastive learning:

```bash
cd CNN

# Train with default configuration
python training/train.py --config ../configs/training_config.yaml

# Custom training
python training/train.py \
    --data_dir ../AudioToSpectogram/output_mel \
    --batch_size 128 \
    --epochs 100 \
    --learning_rate 0.001 \
    --temperature 0.5 \
    --embedding_dim 256
```

**Key Hyperparameters:**
- `--batch_size`: Training batch size (default: 128)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--temperature`: Temperature parameter for contrastive loss (default: 0.5)
- `--embedding_dim`: Dimension of output embeddings (default: 256)

### 4. Embedding Extraction

Extract embeddings from trained model:

```bash
cd CNN

# Extract embeddings for all songs
python embeddings/extract_embeddings.py \
    --checkpoint ../checkpoints/best_model.pth \
    --audio_dir ../AudioToSpectogram/fma_small_dataset \
    --output_dir ../embeddings_db \
    --batch_size 32
```

### 5. Song Recommendation

Get song recommendations based on similarity:

```bash
cd CNN

# Get recommendations for a specific song
python recommendation/recommender.py \
    --query_song "path/to/song.mp3" \
    --embeddings_db ../embeddings_db \
    --top_k 10 \
    --model_checkpoint ../checkpoints/best_model.pth
```

**Python API Example:**
```python
from CNN.recommendation.recommender import MusicRecommender

# Initialize recommender
recommender = MusicRecommender(
    model_path='checkpoints/best_model.pth',
    embeddings_db='embeddings_db'
)

# Get recommendations
recommendations = recommender.recommend(
    query_song='path/to/song.mp3',
    top_k=10
)

for rank, (song_path, similarity) in enumerate(recommendations, 1):
    print(f"{rank}. {song_path} (similarity: {similarity:.4f})")
```

---

## 🔬 Technical Details

### Spectrogram Representation

**Mel-Spectrograms** are used because they:
- Mimic human auditory perception
- Reduce dimensionality while preserving musical information
- Work well with CNNs for audio tasks

**Spectrogram Parameters:**
- Sample rate: 22,050 Hz
- Window size: 2048 samples
- Hop length: 512 samples
- Mel bins: 128
- Duration: 30 seconds

### CNN Encoder Architecture

The encoder consists of:
1. **Convolutional blocks**: Extract hierarchical features
   - 4-6 convolutional layers with batch normalization
   - ReLU activation and max pooling
   - Increasing filter sizes: 64 → 128 → 256 → 512
2. **Global average pooling**: Reduce spatial dimensions
3. **Projection head**: Map to embedding space (256-512 dimensions)

### Contrastive Learning

**SimCLR-based approach:**
- **Positive pairs**: Two augmented views of the same song
- **Negative pairs**: Augmented views of different songs
- **NT-Xent Loss**: Normalized Temperature-scaled Cross Entropy Loss

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

Where:
- $z_i$, $z_j$ are embeddings of positive pairs
- $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ (cosine similarity)
- $\tau$ is the temperature parameter
- $N$ is the batch size

### Similarity Search

**Cosine Similarity** measures the angle between embedding vectors:

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \cos(\theta)$$

- Range: [-1, 1], where 1 = identical, -1 = opposite
- Efficient computation with normalized embeddings
- Scalable to large databases with approximate nearest neighbors (ANN)

---

## 📊 Dataset

### Recommended Datasets

1. **FMA (Free Music Archive)**
   - Small: 8,000 tracks (30s clips)
   - Medium: 25,000 tracks
   - Large: 106,574 tracks
   - [Download](https://github.com/mdeff/fma)

2. **GTZAN Genre Collection**
   - 1,000 tracks (10 genres, 100 tracks each)
   - [Download](http://marsyas.info/downloads/datasets.html)

3. **Million Song Dataset**
   - 1,000,000+ songs
   - [Download](http://millionsongdataset.com/)

### Current Dataset
This project uses the **FMA Small** dataset (located in `AudioToSpectogram/fma_small_dataset/`).

---

## 📈 Results

### Expected Outcomes

After training for 100 epochs:
- **Embedding quality**: Clear clustering of similar songs
- **Recommendation accuracy**: High precision@k for similar songs
- **Training loss**: Converges to ~0.5-1.0
- **Inference speed**: <100ms per song

### Evaluation Metrics

- **Precision@K**: Percentage of relevant songs in top-K recommendations
- **Recall@K**: Coverage of relevant songs in top-K
- **Mean Reciprocal Rank (MRR)**: Quality of ranking
- **t-SNE visualization**: Embedding space visualization

---

## 🔮 Future Improvements

### Short-term
- [ ] Implement FAISS for faster similarity search
- [ ] Add genre classification head
- [ ] Experiment with different encoder architectures (ResNet, EfficientNet)
- [ ] Add more augmentation techniques

### Long-term
- [ ] Multi-modal learning (audio + lyrics + metadata)
- [ ] Temporal modeling with RNNs/Transformers
- [ ] Online learning for continuous improvement
- [ ] Web-based demo interface
- [ ] Mobile app integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Audio conversion code adapted from [AudioToSpectogram](https://github.com/hdnh2006/AudioToSpectogram)
- SimCLR paper: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- SpecAugment: [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779)

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Music Coding! 🎵🎶**
