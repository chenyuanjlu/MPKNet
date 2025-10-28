"""
MRNet模型 - 用于医学影像多标签分类
改编自斯坦福MRNet，适配3D医学影像和多标签分类
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ==================== 原始MRNet (2D切片版本) ====================
class MRNet2D(nn.Module):
    """
    原始MRNet架构
    输入：2D切片序列 [B, num_slices, C, H, W]
    输出：标签预测
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
            num_classes: 输出类别数
            backbone: 骨干网络 ('alexnet', 'resnet18', 'resnet34')
            pretrained: 是否使用ImageNet预训练
            dropout: Dropout概率
        """
        super().__init__()
        
        # 选择骨干网络
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
        
        # 将所有切片展平处理
        x = x.view(-1, *x.shape[2:])  # [B*num_slices, C, H, W]
        
        # 特征提取
        features = self.feature_extractor(x)  # [B*num_slices, feature_dim, H', W']
        
        # 全局平均池化
        features = self.avg_pool(features)  # [B*num_slices, feature_dim, 1, 1]
        features = features.view(batch_size, num_slices, -1)  # [B, num_slices, feature_dim]
        
        # Max pooling聚合所有切片（MRNet核心）
        features = features.max(dim=1)[0]  # [B, feature_dim]
        
        # 分类
        features = self.dropout(features)
        logits = self.fc(features)  # [B, num_classes]
        
        return logits


# ==================== 3D体积版本的MRNet ====================
class MRNet3D(nn.Module):
    """
    MRNet改编为3D版本
    输入：3D体积 [B, C, D, H, W]
    核心思想：沿某个轴提取切片 → 独立处理 → Max Pooling聚合
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        slice_axis: int = 2,  # 0=深度, 1=高度, 2=宽度
        backbone_2d: str = 'resnet18',
        pretrained: bool = False,
        dropout: float = 0.5,
    ):
        """
        Args:
            spatial_dims: 必须为3
            in_channels: 输入通道数
            out_channels: 输出类别数
            slice_axis: 沿哪个轴提取切片 (0, 1, 2)
            backbone_2d: 2D骨干网络
            pretrained: 是否使用预训练（注意：3D医学影像通常为单通道）
            dropout: Dropout概率
        """
        super().__init__()
        
        assert spatial_dims == 3, "MRNet3D only supports 3D input"
        self.slice_axis = slice_axis
        
        # 如果输入是单通道，需要转换为3通道以适配ImageNet预训练
        self.channel_adapter = None
        if in_channels == 1 and pretrained:
            self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        # 2D骨干网络
        if backbone_2d == 'alexnet':
            backbone = models.alexnet(pretrained=pretrained)
            self.feature_extractor = backbone.features
            feature_dim = 256
        elif backbone_2d == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            # 如果是单通道且不用预训练，修改第一层
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
        """
        Args:
            x: [B, C, D, H, W]
        
        Returns:
            logits: [B, out_channels]
        """
        batch_size = x.shape[0]
        
        # 沿指定轴提取切片
        if self.slice_axis == 0:
            # 沿深度轴：[B, C, D, H, W] -> [B, D, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)
            num_slices = x.shape[1]
            x = x.contiguous().view(-1, *x.shape[2:])  # [B*D, C, H, W]
        elif self.slice_axis == 1:
            # 沿高度轴：[B, C, D, H, W] -> [B, H, C, D, W]
            x = x.permute(0, 3, 1, 2, 4)
            num_slices = x.shape[1]
            x = x.contiguous().view(-1, *x.shape[2:])  # [B*H, C, D, W]
        elif self.slice_axis == 2:
            # 沿宽度轴：[B, C, D, H, W] -> [B, W, C, D, H]
            x = x.permute(0, 4, 1, 2, 3)
            num_slices = x.shape[1]
            x = x.contiguous().view(-1, *x.shape[2:])  # [B*W, C, D, H]
        
        # 通道适配（如果需要）
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)  # [B*num_slices, 3, H, W]
        
        # 特征提取
        features = self.feature_extractor(x)  # [B*num_slices, feature_dim, H', W']
        
        # 全局平均池化
        features = self.avg_pool(features).squeeze(-1).squeeze(-1)  # [B*num_slices, feature_dim]
        features = features.view(batch_size, num_slices, -1)  # [B, num_slices, feature_dim]
        
        # 🌟 核心：Max pooling聚合（MRNet的精髓）
        features = features.max(dim=1)[0]  # [B, feature_dim]
        
        # 分类
        features = self.dropout(features)
        logits = self.fc(features)  # [B, out_channels]
        
        return logits


# ==================== 纯3D卷积版本 ====================
class MRNet3DConv(nn.Module):
    """
    使用3D卷积的MRNet变体
    不依赖2D预训练权重，完全从头训练
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # 3D特征提取器（类似AlexNet的结构）
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


# ==================== 轻量级版本（推荐用于小数据集）====================
class LightMRNet(nn.Module):
    """
    轻量级MRNet
    专为小数据集设计（如您的760样本）
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # 简化的3D特征提取
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