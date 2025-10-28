"""
CoPAS模型 - Co-Plane Attention System
改编自原始CoPAS，适配单/多平面医学影像多标签分类
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== 3D ResNet Encoder ====================
class BasicBlock3D(nn.Module):
    """3D ResNet基础块"""
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
    3D ResNet特征提取器
    """
    def __init__(
        self,
        in_channels: int = 1,
        depth: int = 18,
        num_classes: int = 4,
    ):
        super().__init__()
        
        self.inplanes = 64
        
        # 初始卷积
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层配置
        if depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock3D
            self.feature_channel = 512
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        # 构建ResNet层
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
            squeeze_to_vector: 是否压缩为向量
        
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
            # 压缩为向量
            x = F.adaptive_max_pool3d(x, 1)
            x = torch.flatten(x, start_dim=1)  # [B, 512]
        else:
            # 保留深度维度
            x = x.transpose(1, 2)  # [B, D, C, H, W]
            x = F.adaptive_max_pool3d(x, (self.feature_channel, 1, 1))
            x = torch.flatten(x, start_dim=2)  # [B, D, feature_channel]
        
        return x


# ==================== Co-Plane Attention ====================
class CoPlaneAttention(nn.Module):
    """
    协同平面注意力模块
    
    核心思想：使用其他两个平面的信息来增强主平面特征
    例如：矢状位特征 ← attend to → 冠状位特征 & 轴位特征
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.emb_dim = embed_dim
        
        # Query, Key, Value投影
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
            main_f: 主平面特征 [B, D, C]
            co_f1: 协同平面1特征 [B, D, C]
            co_f2: 协同平面2特征 [B, D, C]
        
        Returns:
            融合后的特征 [B, C]
        """
        res = main_f
        
        # 计算Q, K, V
        q = self.mq(main_f)  # [B, D, C]
        k1 = self.mk1(co_f1).permute(0, 2, 1)  # [B, C, D]
        k2 = self.mk2(co_f2).permute(0, 2, 1)  # [B, C, D]
        v1 = self.mv1(co_f1)  # [B, D, C]
        v2 = self.mv2(co_f2)  # [B, D, C]
        
        # 计算注意力权重
        att1 = torch.matmul(q, k1) / np.sqrt(self.emb_dim)  # [B, D, D]
        att1 = torch.softmax(att1, dim=-1)
        
        att2 = torch.matmul(q, k2) / np.sqrt(self.emb_dim)  # [B, D, D]
        att2 = torch.softmax(att2, dim=-1)
        
        # 加权聚合
        out1 = torch.matmul(att1, v1)  # [B, D, C]
        out2 = torch.matmul(att2, v2)  # [B, D, C]
        
        # 残差连接 + LayerNorm
        f = self.norm(0.5 * (out1 + out2) + res)  # [B, D, C]
        
        # Max pooling聚合深度维度
        f = f.transpose(1, 2)  # [B, C, D]
        f = F.adaptive_max_pool1d(f, 1)  # [B, C, 1]
        f = torch.flatten(f, start_dim=1)  # [B, C]
        
        return f


# ==================== Cross-Modal Attention ====================
class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    
    核心思想：融合不同模态（如PDW和T2W）的信息
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
            pdw_f: PDW模态特征 [B, C]
            aux_f: 辅助模态特征(T2W/T1W) [B, C]
        
        Returns:
            融合特征 [B, C]
        """
        # 加性融合
        add_f = pdw_f + aux_f
        
        # 学习注意力权重
        sub_f = torch.cat((pdw_f, aux_f), dim=1)  # [B, 2C]
        att_f = self.transform_matrix(sub_f)  # [B, C]
        att_f = torch.relu(att_f)
        att_f = torch.softmax(att_f, dim=-1)  # [B, C]
        
        # 加权融合
        f = add_f * att_f
        return f


# ==================== 单平面CoPAS（适配您的项目）====================
class SinglePlaneCoPAS(nn.Module):
    """
    单平面CoPAS
    使用协同平面注意力思想，但通过数据增强模拟其他平面
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
        
        # 3D ResNet编码器
        self.encoder = ResNet3DEncoder(in_channels, depth, out_channels)
        
        # 协同平面注意力（如果启用）
        if use_co_attention:
            self.co_plane_att = CoPlaneAttention(self.encoder.feature_channel)
        
        # 分类器
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
            # 简单模式：直接编码+分类
            features = self.encoder(x, squeeze_to_vector=True)
        else:
            # Co-Plane Attention模式
            # 主平面特征
            main_f = self.encoder(x, squeeze_to_vector=False)  # [B, D, C]
            
            # 模拟其他平面（通过旋转/转置）
            # 这里可以通过数据增强或实际提供其他平面数据
            # 为简化，我们使用转置模拟
            with torch.no_grad():
                # 模拟平面1：转置D和W维度
                x_plane1 = x.transpose(2, 4)
                co_f1 = self.encoder(x_plane1, squeeze_to_vector=False)
                
                # 模拟平面2：旋转
                x_plane2 = torch.rot90(x.transpose(2, 4), k=1, dims=[3, 4])
                co_f2 = self.encoder(x_plane2, squeeze_to_vector=False)
            
            # 协同平面注意力
            features = self.co_plane_att(main_f, co_f1, co_f2)
        
        # 分类
        logits = self.classifier(features)
        return logits


# ==================== 轻量级版本（推荐小数据集）====================
class LightCoPAS(nn.Module):
    """
    轻量级CoPAS
    简化版本，适合小数据集（如760样本）
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # 简化的3D编码器
        self.encoder = nn.Sequential(
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
        
        # 自注意力层（简化版Co-Plane Attention）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=base_channels * 4,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(base_channels * 4)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码
        features = self.encoder(x)  # [B, C, D, H, W]
        
        # 重塑为序列
        b, c, d, h, w = features.shape
        features_seq = features.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)  # [B, D*H*W, C]
        
        # 自注意力
        att_out, _ = self.self_attention(features_seq, features_seq, features_seq)
        att_out = self.norm(att_out + features_seq)
        
        # 重塑回3D
        att_out = att_out.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        
        # 分类
        logits = self.classifier(att_out)
        return logits