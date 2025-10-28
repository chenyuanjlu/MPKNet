# Pretrained Models

This directory contains trained models for different ligament labels, used for ensemble prediction.

## Model Files

Place your trained models here with the following naming convention:

- `training_ACL/` - Anterior Cruciate Ligament model
- `training_PCL/` - Posterior Cruciate Ligament model  
- `training_MCL/` - Medial Collateral Ligament model
- `training_LCL/` - Lateral Collateral Ligament model

## Usage

The ensemble model loader (`pack_ensemble_model.py`) reads models from these directories for multi-label inference.

## Note

Model files (`.pth`) are not included in the repository due to file size limitations.