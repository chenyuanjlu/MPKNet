"""
Training Pipeline for Knee Ligament Injury Classification
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from preprocess import (
    load_data_from_csv, 
    split_data, 
    get_train_transforms,
    get_val_transforms,
    MultiLabelDataset
)
from loss import MultiLabelFocalLoss, AsymmetricLoss, create_criterion
from utils import calculate_multilabel_metrics, save_best_model_and_visualizations, calculate_multilabel_metrics_adaptive
from network import DenseNet169, create_multiplane_densenet
from torch.utils.data import WeightedRandomSampler


def create_balanced_sampler(train_files, label_columns):
    """Create weighted sampler"""
    labels = np.array([item['label'] for item in train_files])
    
    if len(label_columns) == 1:
        label_values = labels[:, 0]
        pos_count = int(label_values.sum())
        neg_count = int(len(label_values) - pos_count)
        
        pos_weight = neg_count
        neg_weight = pos_count
        
        total_weight = pos_count * pos_weight + neg_count * neg_weight
        pos_weight_norm = pos_weight / total_weight * len(label_values)
        neg_weight_norm = neg_weight / total_weight * len(label_values)
        
        weights = np.where(label_values == 1, pos_weight_norm, neg_weight_norm)
        
        print(f"   Positive samples: {pos_count}")
        print(f"   Negative samples: {neg_count}")
        
    else:
        # Multi-label mode
        num_positives = labels.sum(axis=1)
        max_positives = num_positives.max()
        weights = (num_positives + 1) / (max_positives + 1)
        
        print(f"\n📊 Balanced Sampling (multi-label):")
        print(f"   Samples with 0 positives: {(num_positives == 0).sum()}")
        print(f"   Samples with 1+ positives: {(num_positives > 0).sum()}")
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                label_names, num_planes=1):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    all_preds = []
    all_labels = []
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle multi-plane input
        if num_planes == 1:
            inputs = batch_data["img"].to(device)
        else:
            inputs = torch.stack([
                batch_data[f"img{i}"].squeeze(1).to(device) for i in range(num_planes)
            ], dim=1)
        
        labels = batch_data["label"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Collect predictions and labels
        with torch.no_grad():
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = epoch_loss / len(train_loader)
    
    # Calculate training metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = calculate_multilabel_metrics(all_preds, all_labels, label_names=label_names)
    
    return avg_loss, metrics


def validate_epoch(model, val_loader, criterion, device, label_names, num_planes=1):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle multi-plane input
            if num_planes == 1:
                inputs = batch_data["img"].to(device)
            else:
                inputs = torch.stack([
                    batch_data[f"img{i}"].squeeze(1).to(device) for i in range(num_planes)
                ], dim=1)
            
            labels = batch_data["label"].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            
            # Collect predictions and labels
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = epoch_loss / len(val_loader)
    
    # Calculate validation metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = calculate_multilabel_metrics(all_preds, all_labels, label_names=label_names)
    
    # Use adaptive thresholding
    # metrics = calculate_multilabel_metrics_adaptive(
    #     y_pred=all_preds,
    #     y_true=all_labels,
    #     label_names=label_names
    # )
    
    return avg_loss, metrics, all_preds, all_labels


def run_training(config):
    """
    Main training pipeline
    
    Args:
        config: Configuration dictionary
    
    Returns:
        tuple: (trained_model, training_log)
    """
    
    # Get label columns from config
    label_columns = config['data'].get('label_columns', ['ACL', 'PCL', 'MCL', 'LCL'])
    num_classes = len(label_columns)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join("experiments", f"training_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_save_dir}")
    print(f"Classification labels: {label_columns}")
    
    # Load data
    print("=" * 60)
    print("Loading data...")
    
    # Check if using modality augmentation
    if config['data'].get('use_modality_augmentation', False):
        from preprocess import load_data_with_modality_augmentation

        print("Using modality augmentation strategy")
        data_list = load_data_with_modality_augmentation(
            csv_path=config['data']['labels_csv'],
            data_dir=config['data']['images_dir'],
            base_sequences=config['data'].get('base_sequences', ['Sag_PDW']),
            augment_sequences=config['data'].get('augment_sequences', ['Sag_T1W']),
            label_columns=label_columns
        )
    else:
        data_list = load_data_from_csv(
            csv_path=config['data']['labels_csv'],
            data_dir=config['data']['images_dir'],
            sequences=config['data']['sequences'],
            label_columns=label_columns
        )
      
    # Split dataset
    train_files, val_files = split_data(
        data_list,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        label_columns=label_columns
    )
    
    # Get number of planes
    num_planes = len(config['data']['sequences'])
    
    # Create data transforms
    train_transforms = get_train_transforms(
        spatial_size=config['data']['spatial_size'],
        augment_level=config['training']['data_augment'],
        num_planes=num_planes
    )
    val_transforms = get_val_transforms(
        spatial_size=config['data']['spatial_size'],
        num_planes=num_planes
    )
    
    # Create datasets
    train_ds = MultiLabelDataset(data_list=train_files, transform=train_transforms)
    val_ds = MultiLabelDataset(data_list=val_files, transform=val_transforms)
    
    
    # balanced sampling
    use_balanced_sampling = config['training'].get('use_balanced_sampling', False)

    if use_balanced_sampling:
        sampler = create_balanced_sampler(train_files, label_columns)
        train_loader = DataLoader(
            train_ds,
            batch_size=config['training']['batch_size'],
            sampler=sampler,  
            num_workers=config['data']['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        print("✅ Using balanced sampling for training")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        print("✅ Using standard random sampling")
    
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("=" * 60)
    print("Creating model...")
    
    model_type = config['model'].get('model_type', 'DenseNet169')
    
    if model_type == 'DenseNet169':
        if num_planes == 1:
            # Single plane: use original DenseNet169
            model = DenseNet169(
                spatial_dims=3,
                in_channels=1,
                out_channels=num_classes,
            ).to(device)
            print(f"Model: DenseNet169 (single-plane)")
        else:
            # Multi-plane: use multi-plane DenseNet
            fusion_type = config['model'].get('fusion_type', 'concat')
            shared_encoder = config['model'].get('shared_encoder', False)
            
            model = create_multiplane_densenet(
                sequences=config['data']['sequences'],
                out_channels=num_classes,
                fusion_type=fusion_type,
                shared_encoder=shared_encoder
            ).to(device)
            
            print(f"Model: MultiPlaneDenseNet")
            
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    loss_type = config['training']['loss_function']
    criterion = create_criterion(loss_type, label_columns, train_files, device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-7
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(model_save_dir, "tensorboard"))
    
    # Training loop
    print("=" * 60)
    print("Starting training...")
    best_avg_auc = 0.0
    best_epoch = 0
    
    training_log = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "epochs": [],
        "best_metrics": {}
    }
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 60)
        
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            label_names=label_columns,
            num_planes=num_planes
        )
        
        # Validate
        val_loss, val_metrics, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device,
            label_names=label_columns,
            num_planes=num_planes
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_duration = time.time() - epoch_start_time
        
        # Print results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Avg AUC: {train_metrics['avg_auc']:.4f}, Avg F1: {train_metrics['avg_f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Avg AUC: {val_metrics['avg_auc']:.4f}, Avg F1: {val_metrics['avg_f1']:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Duration: {epoch_duration:.2f}s")
        
        # Detailed per-label metrics
        print("\n  Validation metrics per label:")
        for label_name in label_columns:
            metrics = val_metrics['per_label'][label_name]
            print(f"    {label_name}: AUC={metrics['auc']:.4f}, "
                  f"ACC={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, "
                  f"Threshold={metrics['optimal_threshold']:.2f}")
            
        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("AUC/train", train_metrics['avg_auc'], epoch)
        writer.add_scalar("AUC/val", val_metrics['avg_auc'], epoch)
        writer.add_scalar("F1/val", val_metrics['avg_f1'], epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        # Save log
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "learning_rate": current_lr,
            "duration": epoch_duration
        }
        training_log["epochs"].append(epoch_record)
        
        # Save best model and generate visualizations
        if val_metrics['avg_auc'] > best_avg_auc:
            best_avg_auc = val_metrics['avg_auc']
            best_epoch = epoch + 1
            
            training_log["best_metrics"] = save_best_model_and_visualizations(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_metrics=val_metrics,
                val_preds=val_preds,
                val_labels=val_labels,
                label_columns=label_columns,
                model_save_dir=model_save_dir,
                val_loader=val_loader,
                device=device,
                best_avg_auc=best_avg_auc,
                config=config
            )
        
        # Save training log
        log_filename = os.path.join(model_save_dir, "training_log.json")
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    # Training complete
    training_log["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best Avg AUC: {best_avg_auc:.4f} (Epoch {best_epoch})")
    print(f"Model saved at: {model_save_dir}")
    
    writer.close()
    
    return model, training_log