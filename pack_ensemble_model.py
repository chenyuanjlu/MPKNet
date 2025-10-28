"""
Pack Ensemble Model

This script loads 4 independent binary classifiers and packs them
into a single ensemble model file for easier deployment.
"""

import argparse
import torch
import os
import glob
from pathlib import Path


DEFAULT_PRETRAINED_DIR = './pretrained'
DEFAULT_OUTPUT = './pretrained/ensemble_model.pth'


def detect_model_info(checkpoint):
    """Detect model structure from checkpoint"""
    state_dict = checkpoint['model_state_dict']
    
    # Check if multiplane
    is_multiplane = any('plane_encoders' in key for key in state_dict.keys())
    
    if is_multiplane:
        # Count planes
        num_planes = max([int(key.split('.')[1]) for key in state_dict.keys() 
                         if key.startswith('plane_encoders.')]) + 1
        # Infer sequences
        if num_planes == 1:
            sequences = ['Sag_PDW']
        elif num_planes == 2:
            sequences = ['Sag_PDW', 'Cor_PDW']
        else:
            sequences = ['Sag_PDW'] * num_planes
    else:
        num_planes = 1
        sequences = ['Sag_PDW']
    
    return {
        'num_planes': num_planes,
        'sequences': sequences,
        'is_multiplane': is_multiplane
    }


def pack_ensemble_model(pretrained_dir, output_path):
    """Pack 4 binary models into a single ensemble model file"""
    
    print("="*70)
    print("Packing Ensemble Model")
    print("="*70)
    
    label_names = ['ACL', 'PCL', 'MCL', 'LCL']
    packed_data = {
        'label_names': label_names,
        'models': {}
    }
    
    # Load each model
    for label in label_names:
        # Find model folder
        pattern = os.path.join(pretrained_dir, f'*_{label}')
        matching_folders = glob.glob(pattern)
        
        if not matching_folders:
            raise FileNotFoundError(f"No model folder found for {label}")
        
        model_path = os.path.join(matching_folders[0], 'best_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        print(f"\n📦 Loading {label} model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Detect model info
        model_info = detect_model_info(checkpoint)
        print(f"   Planes: {model_info['num_planes']}, Sequences: {model_info['sequences']}")
        
        # Save model data
        packed_data['models'][label] = {
            'state_dict': checkpoint['model_state_dict'],
            'num_planes': model_info['num_planes'],
            'sequences': model_info['sequences'],
            'is_multiplane': model_info['is_multiplane']
        }
    
    # Save packed model
    torch.save(packed_data, output_path)
    
    # Print summary
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n✅ Ensemble model packed successfully!")
    print(f"   Output: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"\n📊 Model summary:")
    for label in label_names:
        info = packed_data['models'][label]
        print(f"   {label}: {info['num_planes']} plane(s) - {info['sequences']}")


def main():
    parser = argparse.ArgumentParser(
        description='Pack ensemble model from individual binary classifiers'
    )
    
    parser.add_argument(
        '--pretrained_dir',
        type=str,
        default=DEFAULT_PRETRAINED_DIR,
        help=f'Directory containing model folders (default: {DEFAULT_PRETRAINED_DIR})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        help=f'Output path for packed model (default: {DEFAULT_OUTPUT})'
    )
    
    args = parser.parse_args()
    
    pack_ensemble_model(
        pretrained_dir=args.pretrained_dir,
        output_path=args.output
    )


if __name__ == '__main__':
    main()