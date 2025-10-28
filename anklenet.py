"""
AnkleNet - Comparison baseline model

3D Vision Transformer for ankle MRI multi-label classification.
Adapted from: [https://github.com/ChiariRay/AnkleNet.git]
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


class PreNorm(nn.Module):
    """Pre-LayerNorm"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
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


# ==================== 3D  ====================
class Conv3DFeatureExtractor(nn.Module):
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
        
        assert d % p == 0 and h % p == 0 and w % p == 0, \
            f"Image dimensions ({d}, {h}, {w}) must be divisible by patch size {p}"
        
        x = rearrange(x, 'b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', 
                     p1=p, p2=p, p3=p)
        
        # Patch embedding
        x = self.to_patch_embedding(x)
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        return self.dropout(x)

# ==================== AnkleNet ====================
class AnkleNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # 简化的特征提取
        self.feature_extractor = Conv3DFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=3
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
    
