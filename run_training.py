from trainer import run_training
from config import config


def train_all_binary_classifiers(base_config):
    """Train independent binary classifiers for each label"""
    all_labels = base_config['data']['label_columns']
    results = {}
    
    print(f"\n Training {len(all_labels)} binary classifiers: {all_labels}\n")
    
    for idx, label in enumerate(all_labels, 1):
        print(f"[{idx}/{len(all_labels)}] Training {label}...", end=" ")
        
        import copy
        label_config = copy.deepcopy(base_config)
        label_config['data']['label_columns'] = [label]
        
        # Train
        model, training_log = run_training(label_config)
        
        # Store results
        best_metrics = training_log['best_metrics']
        results[label] = {
            'auc': best_metrics['avg_auc'],
            'f1': best_metrics['avg_f1'],
            'per_label': best_metrics['per_label'][label]
        }
        
        print(f"✅ AUC: {results[label]['auc']:.4f}, F1: {results[label]['f1']:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Final Results Summary:")
    print(f"{'='*60}")
    
    for label in all_labels:
        m = results[label]['per_label']
        print(f"{label:4s}: AUC={m['auc']:.4f} F1={m['f1']:.4f} "
              f"Precision={m['precision']:.4f} Recall={m['recall']:.4f}")
    
    avg_auc = sum(r['auc'] for r in results.values()) / len(results)
    avg_f1 = sum(r['f1'] for r in results.values()) / len(results)
    print(f"\nAverage: AUC={avg_auc:.4f}, F1={avg_f1:.4f}")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("Knee Ligament Injury Classification")
    print("="*60)
        
    print(f"\n📋 Mode: Binary (independent models)")
    results = train_all_binary_classifiers(config)

    print("\n✅ Training completed!\n")