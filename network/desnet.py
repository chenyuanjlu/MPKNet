"""
DenseNet Architecture for Medical Image Multi-Label Classification
"""

from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn


class _DenseLayer(nn.Module):
    """Basic DenseNet Layer"""
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        
        out_channels = bn_size * growth_rate
        
        # Select appropriate convolution and dropout types
        if spatial_dims == 2:
            conv_type = nn.Conv2d
            dropout_type = nn.Dropout2d
            norm_type = nn.BatchNorm2d
        elif spatial_dims == 3:
            conv_type = nn.Conv3d
            dropout_type = nn.Dropout3d
            norm_type = nn.BatchNorm3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        self.layers = nn.Sequential()
        
        # 1x1 convolution
        self.layers.add_module("norm1", norm_type(in_channels))
        self.layers.add_module("relu1", nn.ReLU(inplace=True))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        
        # 3x3 convolution
        self.layers.add_module("norm2", norm_type(out_channels))
        self.layers.add_module("relu2", nn.ReLU(inplace=True))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))
        
        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """DenseNet Dense Block"""
    
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(
                spatial_dims, 
                in_channels, 
                growth_rate, 
                bn_size, 
                dropout_prob
            )
            in_channels += growth_rate
            self.add_module(f"denselayer{i + 1}", layer)


class _Transition(nn.Sequential):
    """DenseNet Transition Layer"""
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        
        # Select appropriate convolution and pooling types
        if spatial_dims == 2:
            conv_type = nn.Conv2d
            pool_type = nn.AvgPool2d
            norm_type = nn.BatchNorm2d
        elif spatial_dims == 3:
            conv_type = nn.Conv3d
            pool_type = nn.AvgPool3d
            norm_type = nn.BatchNorm3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        self.add_module("norm", norm_type(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    DenseNet Base Class
    
    Implements the DenseNet architecture for medical image classification.
    Supports both 2D and 3D inputs.
    
    Reference:
        Huang et al. "Densely Connected Convolutional Networks" CVPR 2017
    """
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        bn_size: int = 4,
        dropout_prob: float = 0.0,
        use_attention: bool = True,
        use_improved_classifier: bool = True,
        classifier_dropout: float = 0.5,
    ) -> None:
        """
        Args:
            spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D)
            in_channels: Number of input channels
            out_channels: Number of output classes
            init_features: Number of filters in the first convolution layer
            growth_rate: Growth rate (k in the paper) - number of channels added per layer
            block_config: Number of layers in each dense block
            bn_size: Multiplicative factor for bottleneck layers
            dropout_prob: Dropout probability
            use_attention: Whether to use attention mechanism (reserved for future use)
            use_improved_classifier: Whether to use improved classifier (reserved for future use)
            classifier_dropout: Dropout probability in classifier (reserved for future use)
        """
        super().__init__()
        
        # Select appropriate convolution and pooling types
        if spatial_dims == 2:
            conv_type = nn.Conv2d
            pool_type = nn.MaxPool2d
            avg_pool_type = nn.AdaptiveAvgPool2d
            norm_type = nn.BatchNorm2d
        elif spatial_dims == 3:
            conv_type = nn.Conv3d
            pool_type = nn.MaxPool3d
            avg_pool_type = nn.AdaptiveAvgPool3d
            norm_type = nn.BatchNorm3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        # First convolution layer
        self.features = nn.Sequential(
            OrderedDict([
                ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ("norm0", norm_type(init_features)),
                ("relu0", nn.ReLU(inplace=True)),
                ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
            ])
        )
        
        # Dense blocks and Transition layers
        num_features = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features += num_layers * growth_rate
            
            if i == len(block_config) - 1:
                # Add normalization after the last block
                self.features.add_module("norm5", norm_type(num_features))
            else:
                # Add transition layer after other blocks
                trans = _Transition(
                    spatial_dims, 
                    in_channels=num_features, 
                    out_channels=num_features // 2
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2
        
        # Classification layers
        self.class_layers = nn.Sequential(
            OrderedDict([
                ("relu", nn.ReLU(inplace=True)),
                ("pool", avg_pool_type(1)),
                ("flatten", nn.Flatten(1)),
                ("out", nn.Linear(num_features, out_channels)),
            ])
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [B, C, H, W] (2D) or [B, C, H, W, D] (3D)
        
        Returns:
            Output tensor of shape [B, num_classes]
        """
        x = self.features(x)
        x = self.class_layers(x)
        return x


class DenseNet169(DenseNet):
    """
    DenseNet-169 Model for Medical Image Multi-Label Classification
    
    Pre-configured DenseNet with 169 layers following the architecture:
    - Dense Block 1: 6 layers
    - Dense Block 2: 12 layers
    - Dense Block 3: 32 layers
    - Dense Block 4: 32 layers
    Total: 169 layers
    """
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        dropout_prob: float = 0.3,
        **kwargs,
    ) -> None:
        """
        Args:
            spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D)
            in_channels: Number of input channels (typically 1 for grayscale medical images)
            out_channels: Number of output classes (number of labels in multi-label classification)
            init_features: Initial number of features (default: 64)
            growth_rate: Growth rate (default: 32)
            dropout_prob: Dropout probability (default: 0.3)
            **kwargs: Additional arguments passed to the base DenseNet class
        """
        # DenseNet-169 configuration: (6, 12, 32, 32)
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=(6, 12, 32, 32), 
            bn_size=4,
            dropout_prob=dropout_prob,
            **kwargs,
        )