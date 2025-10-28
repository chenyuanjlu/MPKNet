"""
CoPAS - Baseline Comparison Model
https://github.com/zqiuak/CoPAS
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== 3D ResNet Encoder ====================
class BasicBlock3D(nn.Module):
    """3D ResNet Basic Block"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, 
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, 
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out


class ResNet3DEncoder(nn.Module):
    """
    3D ResNet Feature Extractor
    """
    def __init__(
        self,
        in_channels: int = 1,
        depth: int = 18,
        num_classes: int = 4,
    ):
        super().__init__()
        
        self.inplanes = 64
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layer configuration
        if depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock3D
            self.feature_channel = 512
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        # Build ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self._init_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, squeeze_to_vector=False):
        """
        Args:
            x: [B, C, D, H, W]
            squeeze_to_vector: Whether to compress to vector
        
        Returns:
            features: [B, D, feature_channel] or [B, feature_channel]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 512, D, H, W]
        
        if squeeze_to_vector:
            # Compress to vector
            x = F.adaptive_max_pool3d(x, 1)
            x = torch.flatten(x, start_dim=1)  # [B, 512]
        else:
            # Preserve depth dimension
            x = x.transpose(1, 2)  # [B, D, C, H, W]
            x = F.adaptive_max_pool3d(x, (self.feature_channel, 1, 1))
            x = torch.flatten(x, start_dim=2)  # [B, D, feature_channel]
        
        return x


# ==================== Co-Plane Attention ====================
class CoPlaneAttention(nn.Module):
    """
    Co-Plane Attention Module
    
    Core idea: Use information from two other planes to enhance the main plane features
    Example: Sagittal features ← attend to → Coronal features & Axial features
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.emb_dim = embed_dim
        
        # Query, Key, Value projections
        self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv2 = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.mq, self.mk1, self.mk2, self.mv1, self.mv2]:
            nn.init.xavier_uniform_(m.weight)
    
    def forward(self, main_f, co_f1, co_f2):
        """
        Args:
            main_f: Main plane features [B, D, C]
            co_f1: Co-plane 1 features [B, D, C]
            co_f2: Co-plane 2 features [B, D, C]
        
        Returns:
            Fused features [B, C]
        """
        res = main_f
        
        # Compute Q, K, V
        q = self.mq(main_f)  # [B, D, C]
        k1 = self.mk1(co_f1).permute(0, 2, 1)  # [B, C, D]
        k2 = self.mk2(co_f2).permute(0, 2, 1)  # [B, C, D]
        v1 = self.mv1(co_f1)  # [B, D, C]
        v2 = self.mv2(co_f2)  # [B, D, C]
        
        # Compute attention weights
        att1 = torch.matmul(q, k1) / np.sqrt(self.emb_dim)  # [B, D, D]
        att1 = torch.softmax(att1, dim=-1)
        
        att2 = torch.matmul(q, k2) / np.sqrt(self.emb_dim)  # [B, D, D]
        att2 = torch.softmax(att2, dim=-1)
        
        # Weighted aggregation
        out1 = torch.matmul(att1, v1)  # [B, D, C]
        out2 = torch.matmul(att2, v2)  # [B, D, C]
        
        # Residual connection + LayerNorm
        f = self.norm(0.5 * (out1 + out2) + res)  # [B, D, C]
        
        # Max pooling to aggregate depth dimension
        f = f.transpose(1, 2)  # [B, C, D]
        f = F.adaptive_max_pool1d(f, 1)  # [B, C, 1]
        f = torch.flatten(f, start_dim=1)  # [B, C]
        
        return f


# ==================== Cross-Modal Attention ====================
class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Module
    
    Core idea: Fuse information from different modalities (e.g., PDW and T2W)
    """
    def __init__(self, feature_channel: int = 512):
        super().__init__()
        self.transform_matrix = nn.Linear(2 * feature_channel, feature_channel)
        self.norm = nn.BatchNorm1d(num_features=feature_channel)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.transform_matrix.weight)
        nn.init.constant_(self.transform_matrix.bias, 0)
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)
    
    def forward(self, pdw_f, aux_f):
        """
        Args:
            pdw_f: PDW modality features [B, C]
            aux_f: Auxiliary modality features (T2W/T1W) [B, C]
        
        Returns:
            Fused features [B, C]
        """
        # Additive fusion
        add_f = pdw_f + aux_f
        
        # Learn attention weights
        sub_f = torch.cat((pdw_f, aux_f), dim=1)  # [B, 2C]
        att_f = self.transform_matrix(sub_f)  # [B, C]
        att_f = torch.relu(att_f)
        att_f = torch.softmax(att_f, dim=-1)  # [B, C]
        
        # Weighted fusion
        f = add_f * att_f
        return f


class CoPAS(nn.Module):
    """
    Single-plane CoPAS
    Uses co-plane attention concept by simulating other planes through data augmentation
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        depth: int = 18,
        use_co_attention: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()
        
        assert spatial_dims == 3, "CoPAS only supports 3D input"
        
        self.use_co_attention = use_co_attention
        
        # 3D ResNet encoder
        self.encoder = ResNet3DEncoder(in_channels, depth, out_channels)
        
        # Co-plane attention (if enabled)
        if use_co_attention:
            self.co_plane_att = CoPlaneAttention(self.encoder.feature_channel)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.feature_channel, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        
        Returns:
            logits: [B, out_channels]
        """
        if not self.use_co_attention:
            # Simple mode: direct encoding + classification
            features = self.encoder(x, squeeze_to_vector=True)
        else:
            # Co-Plane Attention mode
            # Main plane features
            main_f = self.encoder(x, squeeze_to_vector=False)  # [B, D, C]
            
            # Simulate other planes (through rotation/transpose)
            # Can be replaced with actual multi-plane data or augmentation
            # For simplicity, we use transpose to simulate
            with torch.no_grad():
                # Simulate plane 1: transpose D and W dimensions
                x_plane1 = x.transpose(2, 4)
                co_f1 = self.encoder(x_plane1, squeeze_to_vector=False)
                
                # Simulate plane 2: rotation
                x_plane2 = torch.rot90(x.transpose(2, 4), k=1, dims=[3, 4])
                co_f2 = self.encoder(x_plane2, squeeze_to_vector=False)
            
            # Co-plane attention
            features = self.co_plane_att(main_f, co_f1, co_f2)
        
        # Classification
        logits = self.classifier(features)
        return logits