"""
Loss functions for multi-label classification
"""

from .multilabel_losses import MultiLabelFocalLoss, AsymmetricLoss
from .criterion_factory import create_criterion

__all__ = [
    'MultiLabelFocalLoss',
    'AsymmetricLoss',
    'create_criterion',
]