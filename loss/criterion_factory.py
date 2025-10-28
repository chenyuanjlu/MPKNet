"""
Criterion Factory for Creating Loss Functions
"""

import torch
import torch.nn as nn
from .multilabel_losses import MultiLabelFocalLoss, AsymmetricLoss


def create_criterion(loss_type, label_columns, train_files, device, config=None):
    """
    Create loss criterion based on configuration
    
    Args:
        loss_type: Default loss function type ('bce', 'focal', 'asymmetric')
        label_columns: List of label names
        train_files: Training data (for calculating weights if needed)
        device: PyTorch device
        config: Full configuration dict (for label-specific settings)
    
    Returns:
        criterion: Loss function
    """
    
    # Single-label mode: use label-specific config
    if config and len(label_columns) == 1:
        label = label_columns[0]
        label_specific = config['training'].get('label_specific', {})
        
        if label in label_specific:
            label_config = label_specific[label]
            loss_type = label_config.get('loss_function', 'bce')
            
            print(f"\n📋 {label} Loss Configuration:")
            
            if loss_type == 'bce':
                criterion = nn.BCEWithLogitsLoss()
                print(f"   BCE (no weight)")
                
            elif loss_type == 'bce_weighted':
                weight = label_config.get('pos_weight', 1.0)
                pos_weight = torch.tensor([weight], dtype=torch.float32).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"   BCE with pos_weight={weight}")
                
            elif loss_type == 'focal':
                alpha = label_config.get('alpha', 0.25)
                gamma = label_config.get('gamma', 2.0)
                criterion = MultiLabelFocalLoss(alpha=alpha, gamma=gamma)
                print(f"   Focal Loss (alpha={alpha}, gamma={gamma})")
                
            elif loss_type == 'asymmetric':
                gamma_neg = label_config.get('gamma_neg', 4)
                gamma_pos = label_config.get('gamma_pos', 1)
                clip = label_config.get('clip', 0.05)
                criterion = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
                print(f"   Asymmetric Loss (gamma_neg={gamma_neg}, gamma_pos={gamma_pos})")
                
            else:
                criterion = nn.BCEWithLogitsLoss()
                print(f"   Default BCE")
            
            return criterion
    
    # Multi-label mode or no specific config: use default
    if loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        print("Loss function: BCE")
        
    elif loss_type == 'focal':
        criterion = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
        print("Loss function: Focal Loss")
        
    elif loss_type == 'asymmetric':
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        print("Loss function: Asymmetric Loss")
        
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss function: Default BCE")
    
    return criterion