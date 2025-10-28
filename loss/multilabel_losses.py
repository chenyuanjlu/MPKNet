"""
Loss functions for multi-label classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label Focal Loss
    Suitable for imbalanced multi-label classification
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive vs negative examples
        gamma: Exponent of the modulating factor (1 - p_t) to focus on hard examples
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for imbalanced multi-label classification
    
    Applies different focusing to positive and negative samples.
    Particularly effective for imbalanced datasets.
    
    Reference:
        Ridnik et al. "Asymmetric Loss For Multi-Label Classification" ICCV 2021
    
    Args:
        gamma_neg: Focusing parameter for negative samples (default: 4)
        gamma_pos: Focusing parameter for positive samples (default: 1)
        clip: Probability margin for negative samples (default: 0.05)
    """
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, inputs, targets):
        # Calculate probabilities
        probs = torch.sigmoid(inputs)
        
        # Positive sample loss
        pos_loss = targets * torch.log(probs.clamp(min=1e-8))
        pos_loss = pos_loss * (1 - probs) ** self.gamma_pos
        
        # Negative sample loss
        neg_probs = (probs - self.clip).clamp(min=0)
        neg_loss = (1 - targets) * torch.log((1 - neg_probs).clamp(min=1e-8))
        neg_loss = neg_loss * neg_probs ** self.gamma_neg
        
        loss = -(pos_loss + neg_loss)
        return loss.mean()