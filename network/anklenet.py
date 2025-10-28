"""
AnkleNet模型 - 用于医学影像多标签分类
改编自原始AnkleNet，适配3D医学影像和单/双平面输入
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ==================== Helper Functions ====================
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


# ==================== 基础组件 ====================
class PreNorm(nn.Module):
    """Pre-LayerNorm"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim=1)
        
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer编码器"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ==================== 3D卷积特征提取器 ====================
class Conv3DFeatureExtractor(nn.Module):
    """
    3D卷积特征提取器
    替代原始的ResNet2D backbone，适配3D医学影像
    """
    def __init__(self, in_channels=1, base_channels=64, num_stages=4):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_stages):
            out_channels = base_channels * (2 ** i)
            
            layers.append(nn.Sequential(
                nn.Conv3d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            ))
            current_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = current_channels
    
    def forward(self, x):
        return self.encoder(x)


class PatchEmbedding3D(nn.Module):
    """
    3D图像到Patch嵌入
    将3D特征图转换为序列tokens
    """
    def __init__(self, in_channels, dim, patch_size=2, dropout=0.):
        super().__init__()
        
        self.patch_size = patch_size
        patch_dim = in_channels * (patch_size ** 3)
        
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, C, D, H, W]
        b, c, d, h, w = x.shape
        p = self.patch_size
        
        # 确保可以整除
        assert d % p == 0 and h % p == 0 and w % p == 0, \
            f"Image dimensions ({d}, {h}, {w}) must be divisible by patch size {p}"
        
        # 重排为patches
        x = rearrange(x, 'b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', 
                     p1=p, p2=p, p3=p)
        
        # Patch embedding
        x = self.to_patch_embedding(x)
        
        # 添加CLS token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 位置编码会在Transformer中自动学习
        return self.dropout(x)


# ==================== 单平面AnkleNet (适配您的框架) ====================
class SinglePlaneAnkleNet(nn.Module):
    """
    单平面AnkleNet
    适用于单一3D医学影像输入（如Sag_PDW）
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 32,
        num_stages: int = 4,
        dim: int = 256,
        depth: int = 3,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 512,
        patch_size: int = 2,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        """
        Args:
            spatial_dims: 空间维度（必须为3）
            in_channels: 输入通道数
            out_channels: 输出类别数
            base_channels: 卷积基础通道数
            num_stages: 卷积阶段数
            dim: Transformer维度
            depth: Transformer深度
            heads: 注意力头数
            dim_head: 每个头的维度
            mlp_dim: MLP隐藏层维度
            patch_size: Patch大小
            dropout: Dropout率
            emb_dropout: Embedding dropout率
        """
        super().__init__()
        
        assert spatial_dims == 3, "SinglePlaneAnkleNet only supports 3D input"
        
        # 3D卷积特征提取
        self.feature_extractor = Conv3DFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages
        )
        
        # Patch嵌入
        self.patch_embedder = PatchEmbedding3D(
            in_channels=self.feature_extractor.out_channels,
            dim=dim,
            patch_size=patch_size,
            dropout=emb_dropout
        )
        
        # Transformer编码器
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        # 特征提取
        features = self.feature_extractor(x)
        
        # Patch嵌入
        tokens = self.patch_embedder(features)
        
        # Transformer编码
        tokens = self.transformer(tokens)
        
        # 使用CLS token进行分类
        cls_token = tokens[:, 0]
        
        # 分类
        logits = self.classifier(cls_token)
        
        return logits


# ==================== 双平面AnkleNet (原始版本的改造) ====================
class DualPlaneAnkleNet(nn.Module):
    """
    双平面AnkleNet
    适用于两个不同平面的3D医学影像输入
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 32,
        num_stages: int = 4,
        dim: int = 256,
        depth: int = 3,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 512,
        patch_size: int = 2,
        cross_attn_depth: int = 2,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        """双平面交叉注意力模型"""
        super().__init__()
        
        # 两个平面的特征提取器（共享权重或独立）
        self.plane1_extractor = Conv3DFeatureExtractor(in_channels, base_channels, num_stages)
        self.plane2_extractor = Conv3DFeatureExtractor(in_channels, base_channels, num_stages)
        
        # Patch嵌入
        self.plane1_embedder = PatchEmbedding3D(
            self.plane1_extractor.out_channels, dim, patch_size, emb_dropout
        )
        self.plane2_embedder = PatchEmbedding3D(
            self.plane2_extractor.out_channels, dim, patch_size, emb_dropout
        )
        
        # 独立的Transformer编码器
        self.plane1_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.plane2_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # 交叉注意力层
        self.cross_attention = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=cross_attn_heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, Attention(dim, heads=cross_attn_heads, dim_head=dim_head, dropout=dropout)),
            ])
            for _ in range(cross_attn_depth)
        ])
        
        # 分类头
        self.plane1_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_channels))
        self.plane2_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_channels))
    
    def forward(self, plane1_img, plane2_img):
        # 特征提取
        plane1_feat = self.plane1_extractor(plane1_img)
        plane2_feat = self.plane2_extractor(plane2_img)
        
        # Patch嵌入
        plane1_tokens = self.plane1_embedder(plane1_feat)
        plane2_tokens = self.plane2_embedder(plane2_feat)
        
        # 独立编码
        plane1_tokens = self.plane1_transformer(plane1_tokens)
        plane2_tokens = self.plane2_transformer(plane2_tokens)
        
        # 交叉注意力
        for plane1_attend_plane2, plane2_attend_plane1 in self.cross_attention:
            plane1_cls, plane1_patches = plane1_tokens[:, :1], plane1_tokens[:, 1:]
            plane2_cls, plane2_patches = plane2_tokens[:, :1], plane2_tokens[:, 1:]
            
            plane1_cls = plane1_attend_plane2(plane1_cls, context=plane2_patches, kv_include_self=True) + plane1_cls
            plane2_cls = plane2_attend_plane1(plane2_cls, context=plane1_patches, kv_include_self=True) + plane2_cls
            
            plane1_tokens = torch.cat((plane1_cls, plane1_patches), dim=1)
            plane2_tokens = torch.cat((plane2_cls, plane2_patches), dim=1)
        
        # 分类
        plane1_logits = self.plane1_classifier(plane1_tokens[:, 0])
        plane2_logits = self.plane2_classifier(plane2_tokens[:, 0])
        
        return plane1_logits + plane2_logits


# ==================== 轻量级版本 ====================
class LightAnkleNet(nn.Module):
    """
    轻量级AnkleNet
    适合小数据集（如您的760样本）
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,  # 更小的基础通道
        dim: int = 128,  # 更小的Transformer维度
        depth: int = 2,  # 更浅的Transformer
        heads: int = 4,  # 更少的注意力头
        dropout: float = 0.2,  # 更高的dropout
    ):
        super().__init__()
        
        # 简化的特征提取
        self.feature_extractor = Conv3DFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=3  # 更少的stage
        )
        
        self.patch_embedder = PatchEmbedding3D(
            in_channels=self.feature_extractor.out_channels,
            dim=dim,
            patch_size=2,
            dropout=dropout
        )
        
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=32,
            mlp_dim=dim * 2,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        tokens = self.patch_embedder(features)
        tokens = self.transformer(tokens)
        logits = self.classifier(tokens[:, 0])
        return logits