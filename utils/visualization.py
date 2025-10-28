"""
Visualization utilities for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os


def plot_roc_curves(y_true, y_pred, label_names, save_path):
    """
    Plot ROC curves for each label
    
    Args:
        y_true: Ground truth labels [num_samples, num_labels]
        y_pred: Predicted probabilities [num_samples, num_labels]
        label_names: List of label names
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, label_name in enumerate(label_names):
        # Check if both classes are present
        if len(np.unique(y_true[:, i])) < 2:
            print(f"  ⚠️  Warning: {label_name} has only one class, skipping ROC curve")
            continue
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{label_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Validation Set', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📈 ROC curves saved")


def plot_confusion_matrices(y_true, y_pred, label_names, save_path, threshold=0.5):
    """
    Plot confusion matrix heatmap for each label
    
    Args:
        y_true: Ground truth labels [num_samples, num_labels]
        y_pred: Predicted probabilities [num_samples, num_labels]
        label_names: List of label names
        save_path: Path to save the figure
        threshold: Binary classification threshold
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    num_labels = len(label_names)
    
    # Create subplots
    fig, axes = plt.subplots(1, num_labels, figsize=(5 * num_labels, 4))
    if num_labels == 1:
        axes = [axes]
    
    for i, (ax, label_name) in enumerate(zip(axes, label_names)):
        # Compute confusion matrix
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        
        # Normalize to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annotations = np.empty_like(cm).astype(str)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                annotations[row, col] = f'{cm[row, col]}\n({cm_percent[row, col]:.1f}%)'
        
        # Plot heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar=True, ax=ax, vmin=0, vmax=cm.max())
        
        ax.set_title(f'{label_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    plt.suptitle('Confusion Matrices - Validation Set', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Confusion matrices saved")


def plot_prediction_distribution(y_true, y_pred, label_names, save_path):
    """
    Plot prediction probability distribution for each label
    
    Args:
        y_true: Ground truth labels [num_samples, num_labels]
        y_pred: Predicted probabilities [num_samples, num_labels]
        label_names: List of label names
        save_path: Path to save the figure
    """
    num_labels = len(label_names)
    fig, axes = plt.subplots(1, num_labels, figsize=(5 * num_labels, 4))
    if num_labels == 1:
        axes = [axes]
    
    for i, (ax, label_name) in enumerate(zip(axes, label_names)):
        # Separate predictions for positive and negative samples
        pos_preds = y_pred[y_true[:, i] == 1, i]
        neg_preds = y_pred[y_true[:, i] == 0, i]
        
        # Plot histograms
        ax.hist(neg_preds, bins=20, alpha=0.6, color='blue', 
               label=f'Negative (n={len(neg_preds)})', density=True)
        ax.hist(pos_preds, bins=20, alpha=0.6, color='red', 
               label=f'Positive (n={len(pos_preds)})', density=True)
        
        # Add threshold line
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, 
                  label='Threshold (0.5)')
        
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{label_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper center', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Prediction Distribution - Validation Set', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Prediction distribution saved")

    
def save_all_validation_heatmaps(y_true, y_pred, val_metrics, label_names, save_dir, epoch, viz_config=None):
    """
    Generate and save all validation visualizations
    """
    if viz_config is None:
        viz_config = {}
    
    viz_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\n🎨 Generating validation visualizations for epoch {epoch + 1}...")
    
    # ROC curves
    if viz_config.get('roc_curves', True):
        roc_path = os.path.join(viz_dir, f"roc_curves_epoch{epoch+1}.png")
        plot_roc_curves(y_true, y_pred, label_names, roc_path)
    
    # Confusion matrices
    if viz_config.get('confusion_matrices', True):
        cm_path = os.path.join(viz_dir, f"confusion_matrices_epoch{epoch+1}.png")
        plot_confusion_matrices(y_true, y_pred, label_names, cm_path)
 
    # Prediction distribution
    if viz_config.get('prediction_distribution', True):
        dist_path = os.path.join(viz_dir, f"prediction_distribution_epoch{epoch+1}.png")
        plot_prediction_distribution(y_true, y_pred, label_names, dist_path)
      
    print(f"✅ All visualizations saved to: {viz_dir}\n")