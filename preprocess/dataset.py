"""
Custom dataset classes
"""

import torch
from monai.data import Dataset


class MultiLabelDataset(Dataset):
    """
    Multi-label classification dataset
    
    Args:
        data_list: List of data dictionaries
        transform: MONAI transforms to apply
    """
    
    def __init__(self, data_list, transform=None):
        super().__init__(data=data_list, transform=transform)
    
    def __getitem__(self, index):
        data_item = self.data[index]
        
        if self.transform:
            transformed = self.transform(data_item)
        else:
            transformed = data_item
        
        transformed['label'] = torch.tensor(data_item['label'], dtype=torch.float32)
        
        return transformed