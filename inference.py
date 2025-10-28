"""
Inference Script for Knee Ligament Injury Classification

"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from preprocess import get_val_transforms
from monai.data import Dataset, DataLoader
from network import DenseNet169, create_multiplane_densenet
from monai.visualize import OcclusionSensitivity
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


# Default configuration
DEFAULT_INPUT_DIR = './test_data'
DEFAULT_MODEL_PATH = './pretrained/ensemble_model.pth'
DEFAULT_OUTPUT_DIR = './results'
DEFAULT_OUTPUT_CSV = 'predictions.csv'


class EnsembleModel(nn.Module):
    """Ensemble of 4 binary classifiers"""
    
    def __init__(self, packed_data, device='cuda'):
        super().__init__()
        
        self.device = device
        self.label_names = packed_data['label_names']
        self.models = nn.ModuleDict()
        
        # Create and load each model
        for label in self.label_names:
            model_data = packed_data['models'][label]
            
            # Create model based on structure
            if model_data['is_multiplane']:
                model = create_multiplane_densenet(
                    sequences=model_data['sequences'],
                    out_channels=1,
                    fusion_type='concat',
                    shared_encoder=False
                )
            else:
                model = DenseNet169(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                )
            
            # Load weights
            model.load_state_dict(model_data['state_dict'])
            model.eval()
            model = model.to(device)
            
            self.models[label] = model
            self.num_planes = model_data['num_planes']
    
    def forward(self, x):
        outputs = []
        for label in self.label_names:
            logits = self.models[label](x)
            outputs.append(logits)
        return torch.cat(outputs, dim=1)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x, thresholds=None):
        probs = self.predict_proba(x)
        
        if thresholds is None:
            thresholds = {label: 0.5 for label in self.label_names}
        elif isinstance(thresholds, (int, float)):
            thresholds = {label: thresholds for label in self.label_names}
        
        predictions = torch.zeros_like(probs)
        for i, label in enumerate(self.label_names):
            threshold = thresholds.get(label, 0.5)
            predictions[:, i] = (probs[:, i] > threshold).float()
        
        return predictions

    
def generate_single_heatmap(occ_sens, input_img, label_idx, label_name, 
                           patient_id, pred_prob, pred_class, save_dir, num_planes):
    """Generate occlusion heatmap for a single sample"""
    try:
        # Compute occlusion sensitivity
        occ_result, _ = occ_sens(x=input_img)
        occ_map = occ_result[0, label_idx]
        
        # Select middle slice
        depth_slice = input_img.shape[-1] // 2
        
        # Extract slice from original image
        if num_planes == 1:
            img_slice = input_img[0, 0, :, :, depth_slice].detach().cpu().numpy()
        else:
            # Use first plane for visualization
            img_slice = input_img[0, 0, :, :, depth_slice].detach().cpu().numpy()
        
        # Calculate corresponding index in occlusion map
        occ_depth_ratio = occ_map.shape[2] / input_img.shape[-1]
        occ_depth_idx = int(depth_slice * occ_depth_ratio)
        occ_slice = occ_map[:, :, occ_depth_idx].detach().cpu().numpy()
        
        # Upsample occlusion map
        zoom_factors = [
            img_slice.shape[0] / occ_slice.shape[0],
            img_slice.shape[1] / occ_slice.shape[1]
        ]
        occ_slice_upsampled = zoom(occ_slice, zoom_factors, order=3)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(img_slice, cmap='gray', aspect='equal')
        axes[0].set_title(f'Original - {patient_id}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img_slice, cmap='gray', aspect='equal')
        im = axes[1].imshow(occ_slice_upsampled, cmap='jet', alpha=0.6, aspect='equal')
        axes[1].set_title(f'{label_name}: Pred={pred_class} (P={pred_prob:.3f})', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        plt.tight_layout()
        save_path = save_dir / f"{patient_id}_{label_name}_heatmap.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️  Failed to generate heatmap for {patient_id} - {label_name}: {e}")
        
        
def find_mri_files(input_dir, sequences):
    """Find MRI files in input directory"""
    data_list = []
    
    sequence_file_map = {
        'Sag_PDW': 'Sag_PDW.nii.gz',
        'Cor_PDW': 'Cor_PDW.nii.gz',
        'Sag_T1W': 'Sag_T1W.nii.gz',
        'Cor_T1W': 'Cor_T1W.nii.gz',
        'Ax_PDW': 'Ax_PDW.nii.gz'
    }
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    for patient_folder in sorted(input_path.iterdir()):
        if not patient_folder.is_dir():
            continue
        
        patient_id = patient_folder.name
        
        if len(sequences) == 1:
            # Single sequence
            seq_file = sequence_file_map.get(sequences[0], sequences[0])
            img_path = patient_folder / seq_file
            if img_path.exists():
                data_list.append({
                    'patient_id': patient_id,
                    'img': str(img_path)
                })
        else:
            # Multi-sequence
            img_paths = []
            all_exist = True
            
            for seq in sequences:
                seq_file = sequence_file_map.get(seq, seq)
                seq_path = patient_folder / seq_file
                if seq_path.exists():
                    img_paths.append(str(seq_path))
                else:
                    all_exist = False
                    break
            
            if all_exist:
                data_dict = {'patient_id': patient_id}
                for idx, path in enumerate(img_paths):
                    data_dict[f'img{idx}'] = path
                data_list.append(data_dict)
    
    return data_list


def run_inference(input_dir, model_path, output_csv, spatial_size=(64, 64, 64),
                 batch_size=4, thresholds=None, device='cuda',
                 generate_heatmaps=False, mask_size=8): 
    
    """Run inference on MRI images"""
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    print("="*70)
    print("Knee Ligament Injury Classification - Inference")
    print("="*70)
    
    # Load packed model
    print(f"\n📦 Loading ensemble model from: {model_path}")
    packed_data = torch.load(model_path, map_location='cpu')
    
    # Get model info (use first model as reference for num_planes)
    first_label = packed_data['label_names'][0]
    num_planes = packed_data['models'][first_label]['num_planes']
    sequences = packed_data['models'][first_label]['sequences']
    
    print(f"   Model trained with: {num_planes} plane(s) - {sequences}")
    
    # Create ensemble
    model = EnsembleModel(packed_data, device=device)
    print(f"   ✅ Models loaded successfully")
    
    # Find MRI files
    print(f"\n📂 Scanning input directory: {input_dir}")
    data_list = find_mri_files(input_dir, sequences)
    print(f"   Found {len(data_list)} patients")
    
    if len(data_list) == 0:
        raise ValueError("No MRI files found")
    
    # Prepare data
    print(f"\n🔄 Preparing data...")
    val_transforms = get_val_transforms(
        spatial_size=spatial_size,
        num_planes=num_planes
    )
    
    dataset = Dataset(data=data_list, transform=val_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    # Run inference
    print(f"\n🚀 Running inference...")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")

    label_names = packed_data['label_names']
    all_predictions = []
    all_probabilities = []

    if generate_heatmaps:
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, 
                               message='.*non-tuple sequence.*')
        
        heatmap_dir = Path(output_csv).parent / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        occ_sens = OcclusionSensitivity(nn_module=model, mask_size=mask_size, n_batch=16)
        print(f"   Heatmap output: {heatmap_dir}")

    model.eval()
    batch_idx = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Processing"):
            # Prepare input
            if num_planes == 1:
                inputs = batch_data["img"].to(device)
            else:
                inputs = torch.stack([
                    batch_data[f"img{i}"].squeeze(1).to(device) 
                    for i in range(num_planes)
                ], dim=1)
            
            # Get predictions
            probs = model.predict_proba(inputs)
            preds = model.predict(inputs, thresholds=thresholds)
            
            all_probabilities.append(probs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            
            # generate_heatmaps
            if generate_heatmaps:
                for sample_idx in range(inputs.shape[0]):
                    patient_idx = batch_idx * batch_size + sample_idx
                    if patient_idx >= len(data_list):
                        break

                    patient_id = data_list[patient_idx]['patient_id']
                    single_input = inputs[sample_idx:sample_idx+1]

                    for label_idx, label_name in enumerate(label_names):
                        pred_prob = probs[sample_idx, label_idx].item()
                        pred_class = preds[sample_idx, label_idx].item()

                        if pred_prob > 0.3:
                            generate_single_heatmap(
                                occ_sens=occ_sens,
                                input_img=single_input,
                                label_idx=label_idx,
                                label_name=label_name,
                                patient_id=patient_id,
                                pred_prob=pred_prob,
                                pred_class=int(pred_class),
                                save_dir=heatmap_dir,
                                num_planes=num_planes
                            )
            batch_idx += 1
                
    # Combine results
    all_probabilities = np.vstack(all_probabilities)
    all_predictions = np.vstack(all_predictions)
    all_patient_ids = [item['patient_id'] for item in data_list]
    
    # Create results DataFrame
    label_names = packed_data['label_names']
    results_df = pd.DataFrame({'Patient_ID': all_patient_ids})
    
    for i, label in enumerate(label_names):
        results_df[f'{label}_Prediction'] = all_predictions[:, i].astype(int)
        results_df[f'{label}_Probability'] = all_probabilities[:, i]
    
    # Save results
    results_df.to_csv(output_csv, index=False)
    
    # Print summary
    print(f"\n✅ Inference complete!")
    print(f"   Total patients: {len(results_df)}")
    print(f"   Results saved to: {output_csv}")
    
    print(f"\n📊 Prediction summary:")
    for label in label_names:
        pos_count = results_df[f'{label}_Prediction'].sum()
        pos_ratio = pos_count / len(results_df) * 100
        print(f"   {label}: {pos_count}/{len(results_df)} ({pos_ratio:.1f}%) positive")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Knee Ligament Injury Classification Inference'
    )
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default=DEFAULT_INPUT_DIR,
        help=f'Directory containing patient folders with MRI images (default: {DEFAULT_INPUT_DIR})'
    )

    parser.add_argument(
        '--model_path', 
        type=str, 
        default=DEFAULT_MODEL_PATH,
        help=f'Path to packed ensemble model (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--output_csv', 
        type=str, 
        default='predictions.csv',
        help='Path to save results (default: predictions.csv)'
    )
    
    parser.add_argument(
        '--spatial_size', 
        type=int, 
        nargs=3,
        default=[64, 64, 64],
        help='Image spatial size (default: 64 64 64)'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=4,
        help='Batch size (default: 4)'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device (default: cuda)'
    )
    
    parser.add_argument(
        '--thresholds', 
        type=str,
        default=None,
        help='JSON string of thresholds, e.g., {"ACL":0.5,"PCL":0.6}'
    )
        
    parser.add_argument(
        '--generate_heatmaps',
        action='store_true',
        default=True,
        help='Generate occlusion sensitivity heatmaps for predictions (default: True)'
    )

    parser.add_argument(
        '--no_heatmaps',
        action='store_false',
        dest='generate_heatmaps',
        help='Disable heatmap generation'
    )
    
    parser.add_argument(
        '--mask_size',
        type=int,
        default=16,
        help='Occlusion mask size for heatmap generation (default: 8)'
    )
    
    args = parser.parse_args()

    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    if not args.output_csv.startswith('./') and not args.output_csv.startswith('/'):
        output_csv = str(output_dir / args.output_csv)
    else:
        output_csv = args.output_csv

    # Parse thresholds
    thresholds = None
    if args.thresholds:
        import json
        thresholds = json.loads(args.thresholds)
    
    # Run inference
    run_inference(
        input_dir=args.input_dir,
        model_path=args.model_path,
        output_csv=output_csv,  
        spatial_size=tuple(args.spatial_size),
        batch_size=args.batch_size,
        thresholds=thresholds,
        device=args.device,
        generate_heatmaps=args.generate_heatmaps,
        mask_size=args.mask_size
    )


if __name__ == '__main__':
    main()