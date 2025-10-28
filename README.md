# MPKNet: Multi-Plane Knee Ligament Injury Classification Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning framework for automated knee ligament injury classification from multi-plane MRI images using DenseNet-based architectures.

## 📋 Overview

MPKNet is designed to classify four types of knee ligament injuries from MRI scans:
- **ACL** (Anterior Cruciate Ligament)
- **PCL** (Posterior Cruciate Ligament)
- **MCL** (Medial Collateral Ligament)
- **LCL** (Lateral Collateral Ligament)

### Key Features

- 🔄 **Multi-Plane Support**: Utilizes multiple MRI planes (Sagittal, Coronal, Axial)
- 🧬 **Multi-Modal Training**: Supports PDW and T1W sequences
- ⚖️ **Class Imbalance Handling**: Balanced sampling and adaptive thresholding
- 🔍 **Model Interpretability**: Occlusion sensitivity heatmap generation
- 🚀 **Ensemble Inference**: Combines multiple binary classifiers

## 🛠️ Installation

### Requirements

- Python >= 3.8
- CUDA >= 11.0 (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MPKNet.git
cd MPKNet

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
MPKNet/
├── config.py                 # Training configuration
├── run_training.py           # Training entry point
├── trainer.py                # Training pipeline
├── inference.py              # Inference script
├── pack_ensemble_model.py    # Model packing utility
├── network/
│   ├── desnet.py            # DenseNet architecture
│   └── densenet_multiplane.py  # Multi-plane DenseNet
├── preprocess/
│   ├── data_loader.py       # Data loading utilities
│   ├── dataset.py           # Dataset classes
│   └── transforms.py        # Data augmentation
├── loss/
│   ├── multilabel_losses.py # Loss functions
│   └── criterion_factory.py # Loss function factory
└── utils/
    ├── metrics.py           # Evaluation metrics
    ├── checkpoint.py        # Model saving/loading
    ├── visualization.py     # Visualization tools
    └── occlusion_sensitivity.py  # Heatmap generation
```

## 📊 Data Preparation

### Dataset Structure

Organize your MRI data in the following structure:

```
data/
├── patient_001/
│   ├── Sag_PDW.nii.gz
│   ├── Cor_PDW.nii.gz
│   ├── Sag_T1W.nii.gz
│   ├── Cor_T1W.nii.gz
│   └── Ax_PDW.nii.gz
├── patient_002/
│   └── ...
└── ...

label.csv  # Labels file
```

### Label Format

CSV file with columns: `patient_id, ACL, PCL, MCL, LCL`

```csv
patient_id,ACL,PCL,MCL,LCL
patient_001,1,0,1,0
patient_002,0,1,0,0
...
```

## 🚀 Quick Start

### 1. Configuration

Edit `config.py` to set your data paths and training parameters:

```python
config = {
    'data': {
        'images_dir': "/path/to/data",
        'labels_csv': "/path/to/label.csv",
        'sequences': ['Sag_PDW', 'Cor_PDW'],  # MRI sequences
        'label_columns': ['ACL'],  # Single label for binary classification
    },
    'training': {
        'epochs': 100,
        'batch_size': 20,
        'learning_rate': 1e-4,
        'loss_function': 'bce',
        'use_balanced_sampling': True,
    }
}
```

### 2. Training

#### Train Binary Classifiers

Train separate models for each ligament:

```bash
# Edit config.py and run
python run_training.py
```

Models will be saved in `experiments/training_YYYYMMDD_HHMMSS_LABEL/`

### 3. Model Packing

Combine trained binary classifiers into a single ensemble:

```bash
python pack_ensemble_model.py \
    --pretrained_dir ./pretrained \
    --output ./pretrained/ensemble_model.pth
```

The script will automatically find model folders named like `training_*_ACL`, `training_*_PCL`, etc.

### 4. Inference

Run inference on new MRI images:

```bash
# Basic inference
python inference.py

# Or with custom paths
python inference.py \
    --input_dir ./test_data \
    --model_path ./pretrained/ensemble_model.pth \
    --output_csv results.csv
```

#### Generate Interpretability Heatmaps

```bash
python inference.py --generate_heatmaps --mask_size 8
```

Results will be saved to:
- `./results/predictions.csv` - Classification results with probabilities
- `./results/heatmaps/` - Occlusion sensitivity heatmaps (if enabled)

### 5. Results Format

Output CSV contains:

```csv
Patient_ID,ACL_Prediction,PCL_Prediction,MCL_Prediction,LCL_Prediction,ACL_Probability,PCL_Probability,MCL_Probability,LCL_Probability
patient_001,1,0,1,0,0.8523,0.3241,0.7854,0.2156
patient_002,0,1,0,0,0.2341,0.8954,0.1234,0.3456
```

## 🎯 Advanced Usage

### Custom Loss Functions

Configure per-label loss functions in `config.py`:

```python
'label_specific': {
    'ACL': {'loss_function': 'bce'},
    'PCL': {'loss_function': 'focal', 'alpha': 0.5, 'gamma': 2.0},
    'MCL': {'loss_function': 'bce_weighted', 'pos_weight': 2.0},
    'LCL': {'loss_function': 'asymmetric', 'gamma_neg': 4, 'gamma_pos': 1}
}
```

Available loss functions:
- `bce`: Standard Binary Cross-Entropy
- `bce_weighted`: BCE with positive class weighting
- `focal`: Focal Loss for hard examples
- `asymmetric`: Asymmetric Loss for imbalanced data

### Balanced Sampling

Enable balanced sampling for imbalanced datasets:

```python
'use_balanced_sampling': True  # Balances positive/negative samples
```

### Multi-Modal Training

Use multiple modalities as data augmentation:

```python
'use_modality_augmentation': True,
'base_sequences': ['Sag_PDW', 'Cor_PDW'],      # Validation sequences
'augment_sequences': ['Sag_T1W', 'Cor_T1W'],   # Training augmentation
```

This treats T1W images as independent training samples while validating only on PDW.

### Data Augmentation Levels

Choose augmentation intensity:

```python
'data_augment': 'basic'  # Options: 'none', 'light', 'basic'
```

### Adaptive Thresholding

The framework automatically finds optimal classification thresholds for each label during validation by maximizing F1 score. Thresholds are displayed in training logs:

```
PCL: AUC=0.7942, ACC=0.8092, F1=0.7500, Precision=0.7000, Recall=0.8000, Threshold=0.75
```

### Visualization Control

Enable/disable specific visualizations:

```python
'visualization': {
    'enable': True,  # Master switch
    'roc_curves': True,
    'confusion_matrices': True,
    'prediction_distribution': True,
}
```

## 📈 Example Results

Validation metrics from trained models:

```
ACL: AUC=0.8839, ACC=0.8421, F1=0.8284, Precision=0.9091, Recall=0.7609, Threshold=0.55
PCL: AUC=0.7942, ACC=0.8092, F1=0.7500, Precision=0.7000, Recall=0.8000, Threshold=0.75
MCL: AUC=0.8199, ACC=0.7895, F1=0.7500, Precision=0.7333, Recall=0.7667, Threshold=0.60
LCL: AUC=0.7975, ACC=0.8421, F1=0.7200, Precision=0.7500, Recall=0.6923, Threshold=0.70
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
'batch_size': 10  # instead of 20

# Or reduce image size
'spatial_size': (48, 48, 48)  # instead of (64, 64, 64)
```

### Class Imbalance Issues
- Enable `use_balanced_sampling: True`
- Use Focal Loss: `'loss_function': 'focal'`
- Adjust pos_weight in BCE: `'pos_weight': 2.0`

### Low Precision / High Recall
- Framework automatically adjusts thresholds
- Check optimal thresholds in training logs
- Adjust if needed in inference with `--thresholds` argument

### Model Not Learning
- Check data distribution in printed statistics
- Verify label format (0/1, not -1/1)
- Increase learning rate or reduce weight decay
- Disable balanced sampling if data is relatively balanced

## 🖥️ System Requirements

### Minimum
- GPU: 8GB VRAM (e.g., RTX 2070)
- RAM: 16GB
- Storage: 50GB for data + models

### Recommended
- GPU: 16GB+ VRAM (e.g., RTX 3090, A100)
- RAM: 32GB+
- Storage: 200GB+ SSD

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{mpknet2024,
  title={MPKNet: Multi-Plane Knee Ligament Injury Classification Network},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MPKNet}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [MONAI](https://monai.io/) medical imaging framework
- DenseNet architecture from [Huang et al., CVPR 2017](https://arxiv.org/abs/1608.06993)
- PyTorch deep learning framework

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your.email@example.com](mailto:your.email@example.com)

## ⚠️ Disclaimer

This project is for **research purposes only** and should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult qualified medical professionals for clinical decisions.

---

**Star ⭐ this repository if you find it helpful!**

