"""
Model checkpoint and visualization utilities
"""

import os
import torch


def save_best_model_and_visualizations(
    model, optimizer, scheduler, epoch, val_metrics, 
    val_preds, val_labels, label_columns, 
    model_save_dir, val_loader, device, best_avg_auc,
    config=None
):
    """
    Save best model checkpoint and generate all visualizations
    """
    
    # Save model checkpoint
    model_save_path = os.path.join(model_save_dir, "best_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_avg_auc': best_avg_auc,
        'val_metrics': val_metrics
    }, model_save_path)
    
    print(f"  ✅ Saved best model! Avg AUC: {best_avg_auc:.4f}")
    
    # Check visualization config
    viz_config = config.get('visualization', {}) if config else {}
    if not viz_config.get('enable', True):
        return {
            "epoch": epoch + 1,
            "avg_auc": best_avg_auc,
            "avg_f1": val_metrics['avg_f1'],
            "exact_match": val_metrics['exact_match'],
            "per_label": val_metrics['per_label']
        }

    # Generate validation visualizations
    if any([
        viz_config.get('roc_curves', True),
        viz_config.get('confusion_matrices', True),
        viz_config.get('prediction_distribution', True),
    ]):
        from .visualization import save_all_validation_heatmaps
        save_all_validation_heatmaps(
            y_true=val_labels,
            y_pred=val_preds,
            val_metrics=val_metrics,
            label_names=label_columns,
            save_dir=model_save_dir,
            epoch=epoch,
            viz_config=viz_config
        )
    
    # Return best metrics for logging
    return {
        "epoch": epoch + 1,
        "avg_auc": best_avg_auc,
        "avg_f1": val_metrics['avg_f1'],
        "exact_match": val_metrics['exact_match'],
        "per_label": val_metrics['per_label']
    }