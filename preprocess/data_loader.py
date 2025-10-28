"""
Data loading and splitting utilities
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data_from_csv(csv_path, data_dir, sequences=['Sag_PDW'], 
                       label_columns=['ACL', 'PCL', 'MCL', 'LCL']):
    """
    Load data from CSV and create data dictionary list
    
    Args:
        csv_path: Path to label.csv file
        data_dir: Path to cb/ folder containing patient data
        sequences: List of sequences to load. Default: ['Sag_PDW']. 
                  Options: ['Sag_PDW', 'Sag_T1W', 'Ax_PDW', 'Cor_PDW', 'Cor_T1W']
        label_columns: List of labels to use. Default: ['ACL', 'PCL', 'MCL', 'LCL']
    
    Returns:
        list: List of data dictionaries, each containing {'img': path_or_paths, 'label': [ACL, PCL, MCL, LCL]}
    """
    
    df = pd.read_csv(csv_path)
    data_list = []
    
    # Map sequence names to file names
    sequence_file_map = {
        'Sag_PDW': 'Sag_PDW.nii.gz',
        'Sag_T1W': 'Sag_T1W.nii.gz',
        'Ax_PDW': 'Ax_PDW.nii.gz',
        'Cor_PDW': 'Cor_PDW.nii.gz',
        'Cor_T1W': 'Cor_T1W.nii.gz'
    }
    
    for idx, row in df.iterrows():
        patient_id = row['ID']
        patient_dir = os.path.join(data_dir, patient_id)
        
        # Check if patient folder exists
        if not os.path.exists(patient_dir):
            continue
        
        # Collect required sequence file paths
        if len(sequences) == 1:
            # Single sequence mode: store single path directly
            seq_file = sequence_file_map[sequences[0]]
            img_path = os.path.join(patient_dir, seq_file)
            
            if os.path.exists(img_path):
                labels = [int(row[col]) for col in label_columns]
                data_list.append({
                    'img': img_path,
                    'label': labels,
                    'patient_id': patient_id
                })
            else:
                print(f"Warning: File not found {img_path}")
        else:
            # Multi-sequence mode: each sequence as independent key
            img_paths = []
            all_exist = True
            
            for idx, seq in enumerate(sequences):
                seq_file = sequence_file_map[seq]
                seq_path = os.path.join(patient_dir, seq_file)
                
                if os.path.exists(seq_path):
                    img_paths.append(seq_path)
                else:
                    print(f"Warning: File not found {seq_path}")
                    all_exist = False
                    break
            
            if all_exist:
                labels = [int(row[col]) for col in label_columns]
                
                # Build multi-plane data dictionary
                data_dict = {'label': labels, 'patient_id': patient_id}
                for idx, path in enumerate(img_paths):
                    data_dict[f'img{idx}'] = path  # img0, img1, img2, ...
                
                data_list.append(data_dict)
    
    print(f"Successfully loaded {len(data_list)} data samples")
    return data_list


def split_data(data_list, test_size=0.2, random_state=42, 
               label_columns=['ACL', 'PCL', 'MCL', 'LCL']):
    """
    Split data into training and validation sets
    
    Args:
        data_list: List of data samples
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        label_columns: List of label names
    
    Returns:
        tuple: (train_files, val_files)
    """
    train_files, val_files = train_test_split(
        data_list, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Training set: {len(train_files)} samples")
    print(f"Validation set: {len(val_files)} samples")
    
    # Training set label distribution
    train_label_dist = np.array([item['label'] for item in train_files])
    
    print("\nTraining set label distribution:")
    for i, name in enumerate(label_columns):
        pos_count = train_label_dist[:, i].sum()
        neg_count = len(train_files) - pos_count
        print(f"  {name}: Positive={pos_count}, Negative={neg_count}, "
              f"Ratio={pos_count/len(train_files):.2%}")
    
    # Validation set label distribution
    val_label_dist = np.array([item['label'] for item in val_files])
    
    print("\nValidation set label distribution:")
    for i, name in enumerate(label_columns):
        pos_count = val_label_dist[:, i].sum()
        neg_count = len(val_files) - pos_count
        print(f"  {name}: Positive={pos_count}, Negative={neg_count}, "
              f"Ratio={pos_count/len(val_files):.2%}")
    
    return train_files, val_files


def load_data_with_modality_augmentation(csv_path, data_dir, 
                                        base_sequences=['Sag_PDW', 'Cor_PDW'],
                                        augment_sequences=['Sag_T1W', 'Cor_T1W'],
                                        label_columns=['ACL', 'PCL', 'MCL', 'LCL']):
    """
    Load data with modality augmentation - treat different modalities as separate samples
    
    This doubles (or more) the training data by treating T1 as independent samples.
    
    Args:
        csv_path: Path to label.csv
        data_dir: Path to data directory
        base_sequences: Base modalities (e.g., PD sequences)
        augment_sequences: Additional modalities to add as separate samples (e.g., T1 sequences)
        label_columns: Label columns
    
    Returns:
        list: Expanded data list with modalities as separate samples
    """
    data_list = []
    
    # Load base sequences
    for seq in base_sequences:
        seq_data = load_data_from_csv(
            csv_path=csv_path,
            data_dir=data_dir,
            sequences=[seq],  # Single sequence
            label_columns=label_columns
        )
        # Add modality tag
        for item in seq_data:
            item['modality'] = seq
        data_list.extend(seq_data)
        print(f"✅ Loaded {len(seq_data)} samples from {seq}")
    
    # Load augmentation sequences
    for seq in augment_sequences:
        seq_data = load_data_from_csv(
            csv_path=csv_path,
            data_dir=data_dir,
            sequences=[seq],
            label_columns=label_columns
        )
        # Add modality tag
        for item in seq_data:
            item['modality'] = seq
        data_list.extend(seq_data)
        print(f"✅ Loaded {len(seq_data)} samples from {seq} (augmentation)")
    
    print(f"\n📊 Total samples after modality augmentation: {len(data_list)}")
    print(f"   Base samples: {len(base_sequences) * len(seq_data)}")
    print(f"   Augmented samples: {len(augment_sequences) * len(seq_data)}")
    
    return data_list