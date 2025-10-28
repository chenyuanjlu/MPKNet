"""
MRNet - Baseline Comparison Model for Medical Image Multi-label Classification
https://github.com/MisaOgura/MRNet.git
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ==================== Original MRNet (2D Slice Version) ====================
class MRNet2D(nn.Module):
    """
    Original MRNet Architecture
    Input: 2D slice series [B, num_slices, C, H, W]
    Output: Label predictions
    """
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = 'alexnet',
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of output classes
            backbone: Backbone network ('alexnet', 'resnet18', 'resnet34')
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability
        """
        super().__init__()
        
        # Select backbone network
        if backbone == 'alexnet':
            self.feature_extractor = models.alexnet(pretrained=pretrained).features
            feature_dim = 256
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: [B, num_slices, C, H, W] - batch of slice series
        
        Returns:
            logits: [B, num_classes]
        """
        batch_size, num_slices = x.shape[0], x.shape[1]
        
        # Flatten all slices for processing
        x = x.view(-1, *x.shape[2:])  # [B*num_slices, C, H, W]
        
        # Feature extraction
        features = self.feature_extractor(x)  # [B*num_slices, feature_dim, H', W']
        
        # Global average pooling
        features = self.avg_pool(features)  # [B*num_slices, feature_dim, 1, 1]
        features = features.view(batch_size, num_slices, -1)  # [B, num_slices, feature_dim]
        
        # Max pooling aggregation across slices (MRNet core idea)
        features = features.max(dim=1)[0]  # [B, feature_dim]
        
        # Classification
        features = self.dropout(features)
        logits = self.fc(features)  # [B, num_classes]
        
        return logits


# ==================== 3D Volume Version of MRNet ====================
class MRNet3D(nn.Module):
    """
    MRNet adapted for 3D volumes
    Input: 3D volume [B, C, D, H, W]
    Core idea: Extract slices along an axis → Process independently → Max pooling aggregation
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        slice_axis: int = 2,  # 0=depth, 1=height, 2=width
        backbone_2d: str = 'resnet18',
        pretrained: bool = False,
        dropout: float = 0.5,
    ):
        
        super().__init__()
        
        assert spatial_dims == 3, "MRNet3D only supports 3D input"
        self.slice_axis = slice_axis
        
        # If input is single-channel, convert to 3-channel for ImageNet pretrained weights
        self.channel_adapter = None
        if in_channels == 1 and pretrained:
            self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        # 2D backbone network
        if backbone_2d == 'alexnet':
            backbone = models.alexnet(pretrained=pretrained)
            self.feature_extractor = backbone.features
            feature_dim = 256
        elif backbone_2d == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            # Modify first layer if single-channel and not pretrained
            if in_channels == 1 and not pretrained:
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
            feature_dim = 512
        elif backbone_2d == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            if in_channels == 1 and not pretrained:
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_2d}")
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feature_dim, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size = x.shape[0]
        
        # Extract slices along specified axis
        if self.slice_axis == 0:
            x = x.permute(0, 2, 1, 3, 4)
            num_slices = x.shape[1]
            x = x.contiguous().view(-1, *x.shape[2:])  # [B*D, C, H, W]
        elif self.slice_axis == 1:
            x = x.permute(0, 3, 1, 2, 4)
            num_slices = x.shape[1]
            x = x.contiguous().view(-1, *x.shape[2:])  # [B*H, C, D, W]
        elif self.slice_axis == 2:
            x = x.permute(0, 4, 1, 2, 3)
            num_slices = x.shape[1]
            x = x.contiguous().view(-1, *x.shape[2:])  # [B*W, C, D, H]
        
        # Channel adaptation (if needed)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)  # [B*num_slices, 3, H, W]
        
        features = self.feature_extractor(x)  # [B*num_slices, feature_dim, H', W']
        features = self.avg_pool(features).squeeze(-1).squeeze(-1)  # [B*num_slices, feature_dim]
        features = features.view(batch_size, num_slices, -1)  # [B, num_slices, feature_dim]
        
        features = features.max(dim=1)[0]  # [B, feature_dim]
        
        features = self.dropout(features)
        logits = self.fc(features)  # [B, out_channels]
        
        return logits


# ==================== Pure 3D Convolution Version ====================
class MRNet3DConv(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # 3D feature extractor (AlexNet-like structure)
        self.feature_extractor = nn.Sequential(
            # Conv1
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            # Conv2
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            # Conv3
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(base_channels * 2, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = self.avg_pool(features).squeeze(-1).squeeze(-1).squeeze(-1)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits


# ==================== Lightweight Version (Recommended for Small Datasets) ====================
class MRNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # Simplified 3D feature extraction
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.classifier(x)
        return x