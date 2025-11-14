"""
CNN Encoder for Audio Spectrograms
Extracts meaningful representations from mel-spectrograms for self-supervised learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock(nn.Module):
    """Convolutional block with Conv2D, BatchNorm, Activation, and Pooling"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1),
                 pool_size: Tuple[int, int] = (2, 2),
                 use_batch_norm: bool = True,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        super().__init__()
        
        layers = []
        
        # Convolutional layer
        layers.append(nn.Conv2d(in_channels, out_channels, 
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding))
        
        # Batch normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'elu':
            layers.append(nn.ELU(inplace=True))
        
        # Pooling
        layers.append(nn.MaxPool2d(kernel_size=pool_size))
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioEncoderCNN(nn.Module):
    """
    CNN Encoder for audio spectrograms.
    Extracts hierarchical features from mel-spectrograms.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        model_config = config.get('model', {})
        input_config = model_config.get('input', {})
        encoder_config = model_config.get('encoder', {})
        
        # Input channels (3 for RGB spectrograms)
        in_channels = input_config.get('channels', 3)
        
        # Encoder settings
        conv_blocks_config = encoder_config.get('conv_blocks', [])
        use_batch_norm = encoder_config.get('use_batch_norm', True)
        activation = encoder_config.get('activation', 'relu')
        pooling_type = encoder_config.get('global_pooling', 'avg')
        
        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        for block_config in conv_blocks_config:
            filters = block_config.get('filters', 64)
            kernel_size = tuple(block_config.get('kernel_size', [3, 3]))
            stride = tuple(block_config.get('stride', [1, 1]))
            padding = tuple(block_config.get('padding', [1, 1]))
            pool_size = tuple(block_config.get('pool_size', [2, 2]))
            dropout = block_config.get('dropout', 0.0)
            
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    pool_size=pool_size,
                    use_batch_norm=use_batch_norm,
                    dropout=dropout,
                    activation=activation
                )
            )
            
            in_channels = filters
        
        # Global pooling
        if pooling_type == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_type == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Store output dimension
        self.out_features = conv_blocks_config[-1].get('filters', 512) if conv_blocks_config else 512
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrogram tensor [B, C, H, W]
        Returns:
            Feature vector [B, out_features]
        """
        # Pass through convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input spectrogram tensor [B, C, H, W]
        Returns:
            List of feature maps from each conv block
        """
        feature_maps = []
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            feature_maps.append(x)
        
        return feature_maps


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps encoder features to embedding space.
    """
    
    def __init__(self, in_features: int, hidden_dim: int = 512, 
                 embedding_dim: int = 256, num_layers: int = 2,
                 use_batch_norm: bool = True, dropout: float = 0.1):
        """
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
            num_layers: Number of layers in projection head
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(in_features, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Additional hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, embedding_dim))
        else:
            layers.append(nn.Linear(in_features, embedding_dim))
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, in_features]
        Returns:
            Projected embeddings [B, embedding_dim]
        """
        return self.projection(x)


class ContrastiveModel(nn.Module):
    """
    Complete model for contrastive learning.
    Combines encoder and projection head.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        model_config = config.get('model', {})
        projection_config = model_config.get('projection_head', {})
        
        # Encoder
        self.encoder = AudioEncoderCNN(config)
        
        # Projection head
        self.projection_head = ProjectionHead(
            in_features=self.encoder.out_features,
            hidden_dim=projection_config.get('hidden_dim', 512),
            embedding_dim=projection_config.get('embedding_dim', 256),
            num_layers=projection_config.get('num_layers', 2),
            use_batch_norm=projection_config.get('use_batch_norm', True),
            dropout=projection_config.get('dropout', 0.1)
        )
        
        # Feature extractor config
        feature_config = model_config.get('feature_extractor', {})
        self.normalize_embeddings = feature_config.get('normalize', True)
        
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass through encoder and projection head.
        
        Args:
            x: Input spectrogram [B, C, H, W]
            return_embedding: If True, return normalized embeddings for inference
        Returns:
            Projected features [B, embedding_dim]
        """
        # Encode
        features = self.encoder(x)
        
        # Project
        embeddings = self.projection_head(features)
        
        # Normalize for inference
        if return_embedding and self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized embeddings (for inference/similarity search).
        
        Args:
            x: Input spectrogram [B, C, H, W]
        Returns:
            Normalized embeddings [B, embedding_dim]
        """
        with torch.no_grad():
            embeddings = self.forward(x, return_embedding=True)
        return embeddings


def build_model(config: dict) -> ContrastiveModel:
    """
    Factory function to build the complete model.
    
    Args:
        config: Model configuration dictionary
    Returns:
        ContrastiveModel instance
    """
    model = ContrastiveModel(config)
    
    # Initialize weights
    init_config = config.get('initialization', {})
    init_method = init_config.get('method', 'kaiming_normal')
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'normal':
                nn.init.normal_(m.weight, mean=0, std=0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    return model
