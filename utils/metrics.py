"""
Evaluation metrics for multi-label classification
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def find_optimal_threshold(y_pred, y_true, metric='f1'):
    """
    Find optimal threshold for binary classification
    
    Args:
        y_pred: Predicted probabilities [num_samples]
        y_true: True binary labels [num_samples]
        metric: Optimization metric ('f1', 'accuracy', 'youden')
    
    Returns:
        float: Optimal threshold
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred_binary = (y_pred > thresh).astype(int)
        
        TP = ((y_pred_binary == 1) & (y_true == 1)).sum()
        TN = ((y_pred_binary == 0) & (y_true == 0)).sum()
        FP = ((y_pred_binary == 1) & (y_true == 0)).sum()
        FN = ((y_pred_binary == 0) & (y_true == 1)).sum()
        
        if metric == 'f1':
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif metric == 'accuracy':
            score = (TP + TN) / (TP + TN + FP + FN)
        elif metric == 'youden':
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            score = sensitivity + specificity - 1
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold


def calculate_multilabel_metrics_adaptive(y_pred, y_true, label_names=['ACL', 'PCL', 'MCL', 'LCL']):
    """Calculate metrics with optimal threshold for each label"""
    
    label_metrics = {}
    
    for i, name in enumerate(label_names):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(y_pred_label, y_true_label, metric='f1')
        y_pred_binary = (y_pred_label > optimal_threshold).astype(int)
        
        # Calculate metrics
        if len(np.unique(y_true_label)) > 1:
            auc = roc_auc_score(y_true_label, y_pred_label)
        else:
            auc = 0.0
        
        acc = accuracy_score(y_true_label, y_pred_binary)
        
        TP = ((y_pred_binary == 1) & (y_true_label == 1)).sum()
        TN = ((y_pred_binary == 0) & (y_true_label == 0)).sum()
        FP = ((y_pred_binary == 1) & (y_true_label == 0)).sum()
        FN = ((y_pred_binary == 0) & (y_true_label == 1)).sum()
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        label_metrics[name] = {
            'auc': float(auc),
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'optimal_threshold': float(optimal_threshold),
            'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)
        }
    
    avg_auc = np.mean([label_metrics[name]['auc'] for name in label_names])
    avg_f1 = np.mean([label_metrics[name]['f1'] for name in label_names])
    
    return {
        'avg_auc': float(avg_auc),
        'avg_f1': float(avg_f1),
        'per_label': label_metrics
    }


def calculate_multilabel_metrics(y_pred, y_true, threshold=0.4, 
                                 label_names=['ACL', 'PCL', 'MCL', 'LCL']):
    """
    Calculate multi-label classification metrics
    
    Args:
        y_pred: Predicted probabilities [num_samples, num_labels]
        y_true: True binary labels [num_samples, num_labels]
        threshold: Threshold for binary classification (default: 0.5)
        label_names: List of label names
    
    Returns:
        dict: Dictionary containing various metrics including:
            - exact_match: Exact match accuracy (all labels correct)
            - avg_auc: Average AUC across all labels
            - avg_f1: Average F1 score across all labels
            - per_label: Per-label metrics (AUC, accuracy, precision, recall, F1)
    """
    
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Overall accuracy (exact match)
    exact_match = (y_pred_binary == y_true).all(axis=1).mean()
    
    # Per-label metrics
    label_metrics = {}
    
    for i, name in enumerate(label_names):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        y_pred_binary_label = y_pred_binary[:, i]
        
        # Calculate AUC (if both positive and negative samples exist)
        if len(np.unique(y_true_label)) > 1:
            auc = roc_auc_score(y_true_label, y_pred_label)
        else:
            auc = 0.0
        
        # Accuracy
        acc = accuracy_score(y_true_label, y_pred_binary_label)
        
        # TP, TN, FP, FN
        TP = ((y_pred_binary_label == 1) & (y_true_label == 1)).sum()
        TN = ((y_pred_binary_label == 0) & (y_true_label == 0)).sum()
        FP = ((y_pred_binary_label == 1) & (y_true_label == 0)).sum()
        FN = ((y_pred_binary_label == 0) & (y_true_label == 1)).sum()
        
        # Precision, Recall, F1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        label_metrics[name] = {
            'auc': float(auc),
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)
        }
    
    # Calculate average metrics
    avg_auc = np.mean([label_metrics[name]['auc'] for name in label_names])
    avg_f1 = np.mean([label_metrics[name]['f1'] for name in label_names])
    
    return {
        'exact_match': float(exact_match),
        'avg_auc': float(avg_auc),
        'avg_f1': float(avg_f1),
        'per_label': label_metrics
    }