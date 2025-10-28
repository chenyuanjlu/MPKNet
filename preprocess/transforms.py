"""
Data augmentation and preprocessing transforms
"""

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, CropForegroundd, Resized, ToTensord,
    RandFlipd, RandRotated, RandZoomd, RandAffined,
    RandShiftIntensityd, RandScaleIntensityd,
    RandGaussianNoised, RandAdjustContrastd, RandHistogramShiftd,
    RandGaussianSmoothd, RandBiasFieldd, RandKSpaceSpikeNoised,
)


def get_train_transforms(spatial_size=(96, 96, 96), augment_level='basic', num_planes=1):
    """
    Training data transforms (with data augmentation)
    
    Args:
        spatial_size: Target spatial size
        augment_level: 'basic' or 'aggressive'
        num_planes: Number of planes (1=single plane, >1=multi-plane)
    
    Returns:
        Composed transforms
    """
    # Set keys based on number of planes
    if num_planes == 1:
        img_keys = ["img"]
    else:
        img_keys = [f"img{i}" for i in range(num_planes)]
    
    basic_transforms = [
        LoadImaged(keys=img_keys, ensure_channel_first=True),
        Spacingd(keys=img_keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True),
        CropForegroundd(keys=img_keys, source_key=img_keys[0]),
        Resized(keys=img_keys, spatial_size=spatial_size),
    ]
    
    # Basic augmentation
    if augment_level == 'basic':
        augmentations = [
            RandFlipd(keys=img_keys, prob=0.5, spatial_axis=0),
            RandRotated(keys=img_keys, prob=0.3, range_x=10, range_y=10, range_z=10),
            RandShiftIntensityd(keys=img_keys, prob=0.4, offsets=0.1),
            RandScaleIntensityd(keys=img_keys, prob=0.4, factors=0.2),
        ]
    # Aggressive augmentation
    elif augment_level == 'aggressive':
        # Medical imaging specific augmentation
        augmentations = [
            # Medical imaging specific augmentation
            RandFlipd(keys=img_keys, prob=0.5, spatial_axis=0),
            RandRotated(keys=img_keys, prob=0.3, range_x=8, range_y=8, range_z=5),
            RandZoomd(keys=img_keys, prob=0.3, min_zoom=0.95, max_zoom=1.05),               
            # Intensity transformations
            RandShiftIntensityd(keys=img_keys, prob=0.5, offsets=0.1),
            RandScaleIntensityd(keys=img_keys, prob=0.5, factors=0.2),
            RandAdjustContrastd(keys=img_keys, prob=0.3, gamma=(0.8, 1.2)),        
        ]
        
    else:
        augmentations = []
    
    final_transforms = [ToTensord(keys=img_keys)]
    
    return Compose(basic_transforms + augmentations + final_transforms)


def get_val_transforms(spatial_size=(96, 96, 96), num_planes=1):  
    """ Validation data transforms (without augmentation) """
    if num_planes == 1:
        img_keys = ["img"]
    else:
        img_keys = [f"img{i}" for i in range(num_planes)]
    
    return Compose([
        LoadImaged(keys=img_keys, ensure_channel_first=True),
        Spacingd(keys=img_keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True),
        CropForegroundd(keys=img_keys, source_key=img_keys[0]),
        Resized(keys=img_keys, spatial_size=spatial_size),
        ToTensord(keys=img_keys),
    ])