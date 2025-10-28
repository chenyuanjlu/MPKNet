"""
Multi-Plane Multi-Modal DenseNet for Medical Image Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from network.desnet import _DenseBlock, _Transition


class DenseNetFeatureExtractor(nn.Module):
    """
    DenseNet Feature Extractor
    Extracts features only, without classification head
    """
    
    def __init__(self, spatial_dims=3, in_channels=1, densenet_config=(6, 12, 32, 32)):
        super().__init__()
        
        init_features = 64
        growth_rate = 32
        bn_size = 4
        dropout_prob = 0.0
        
        # Select convolution and pooling types
        if spatial_dims == 3:
            conv_type = nn.Conv3d
            pool_type = nn.MaxPool3d
            norm_type = nn.BatchNorm3d
        else:
            conv_type = nn.Conv2d
            pool_type = nn.MaxPool2d
            norm_type = nn.BatchNorm2d
        
        # First convolutional layer
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
        for i, num_layers in enumerate(densenet_config):
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
            
            if i == len(densenet_config) - 1:
                # Add norm after the last block
                self.features.add_module("norm5", norm_type(num_features))
            else:
                # Add transition after other blocks
                trans = _Transition(
                    spatial_dims, 
                    in_channels=num_features, 
                    out_channels=num_features // 2
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2
        
        self.feature_dim = num_features
        
        # Global pooling
        if spatial_dims == 3:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x


class MultiPlaneDenseNet(nn.Module):
    """
    Multi-Plane DenseNet - Supports arbitrary number of planes/modalities
    
    Args:
        num_planes: Number of planes (automatically inferred from sequences)
        spatial_dims: Spatial dimensions (2D or 3D)
        in_channels: Input channels per plane (typically 1)
        out_channels: Number of output classes
        fusion_type: 'concat' | 'mean' | 'max'
        shared_encoder: Whether to share encoder weights across planes
        dropout_prob: Dropout probability
        densenet_config: DenseNet block configuration (default: DenseNet169)
    """
    
    def __init__(
        self,
        num_planes: int = 1,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        fusion_type: str = 'concat',
        shared_encoder: bool = False,
        dropout_prob: float = 0.3,
        densenet_config: tuple = (6, 12, 32, 32)
    ):
        super().__init__()
        
        self.num_planes = num_planes
        self.fusion_type = fusion_type
        self.shared_encoder = shared_encoder
        
        # Create encoders
        if shared_encoder and num_planes > 1:
            # Shared weights: all planes use the same encoder
            single_encoder = DenseNetFeatureExtractor(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                densenet_config=densenet_config
            )
            self.plane_encoders = nn.ModuleList([single_encoder] * num_planes)
            feature_dim = single_encoder.feature_dim
        else:
            # Independent weights: each plane has its own encoder
            encoders = []
            for _ in range(num_planes):
                encoder = DenseNetFeatureExtractor(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    densenet_config=densenet_config
                )
                encoders.append(encoder)
            self.plane_encoders = nn.ModuleList(encoders)
            feature_dim = encoders[0].feature_dim
        
        self.feature_dim = feature_dim
        
        # Fusion strategy
        if fusion_type == 'concat':
            fused_dim = feature_dim * num_planes
        else:  # mean or max
            fused_dim = feature_dim
        
        # Classification head
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(512, out_channels)
        )
        
        self._init_fusion_layers()
    
    def _init_fusion_layers(self):
        for m in self.fusion_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        """
        Forward pass - Compatible with multiple input formats
        
        Args:
            x: Can be:
               1. Single tensor: [B, 1, H, W, D] (single plane)
               2. Stacked tensor: [B, num_planes, H, W, D] (multi-plane)
               3. Dictionary: {'img0': tensor, 'img1': tensor, ...}
        
        Returns:
            output: [B, out_channels] classification logits
        """
        # Handle different input formats
        if isinstance(x, dict):
            # Dictionary format (MONAI DataLoader format)
            plane_inputs = [x[f'img{i}'] for i in range(self.num_planes)]
        elif isinstance(x, torch.Tensor):
            if self.num_planes == 1:
                plane_inputs = [x]
            else:
                # Multi-plane stacked format
                if x.dim() == 5 and x.shape[1] == self.num_planes:
                    plane_inputs = [x[:, i:i+1, :, :, :] for i in range(self.num_planes)]
                else:
                    raise ValueError(f"Expected [B, {self.num_planes}, H, W, D], got {x.shape}")
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
        
        # Extract features from each plane
        plane_features = []
        for i, plane_input in enumerate(plane_inputs):
            feat = self.plane_encoders[i](plane_input)
            plane_features.append(feat)
        
        # Feature fusion
        if self.fusion_type == 'concat':
            fused_features = torch.cat(plane_features, dim=1)
        elif self.fusion_type == 'mean':
            fused_features = torch.stack(plane_features, dim=1).mean(dim=1)
        elif self.fusion_type == 'max':
            fused_features = torch.stack(plane_features, dim=1).max(dim=1)[0]
        
        # Classification
        output = self.fusion_classifier(fused_features)
        return output


# Convenience constructor function
def create_multiplane_densenet(sequences, out_channels=4, fusion_type='concat', **kwargs):
    """
    Create multi-plane multi-modal DenseNet based on sequence list 
    
    Examples:
        # Single sequence (baseline)
        model = create_multiplane_densenet(['Sag_PDW'], out_channels=2)
        
        # Multi-plane (same modality)
        model = create_multiplane_densenet(['Sag_PDW', 'Cor_PDW', 'Ax_PDW'], out_channels=2)
        
        # Multi-modal (same plane)
        model = create_multiplane_densenet(['Sag_PDW', 'Sag_T1W'], out_channels=2)
    """
    
    num_planes = len(sequences)
    return MultiPlaneDenseNet(
        num_planes=num_planes,
        out_channels=out_channels,
        fusion_type=fusion_type,
        **kwargs
    )