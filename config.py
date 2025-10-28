"""
Training Configuration

Modify parameters here to control experiments.
"""

config = {
    # ==================== Data Configuration ====================
    'data': {
        'images_dir': "./data",
        'labels_csv': "./label.csv",
        
        # Modality augmentation: treat different modalities as separate samples
        'use_modality_augmentation': False,  # Enable multi-modality training
        'base_sequences': ['Sag_PDW'],  # Primary sequences
        'augment_sequences': ['Sag_T1W'],  # Augmentation sequences
               
        'sequences': ['Sag_PDW'],  # MRI sequences to use
        'label_columns': ['ACL'],  # Labels to classify: ['ACL', 'PCL', 'MCL', 'LCL']
        'spatial_size': (64, 64, 64),
        'test_size': 0.2,
        'random_state': 42,
        'num_workers': 8,
    },
    
    # ==================== Model Configuration ====================
     'model': {
        'model_type': 'MultiPlaneDenseNet', #'CoPAS',  #'MRNet', #'AnkleNet', #'MultiPlaneDenseNet', 
        'fusion_type': 'concat',  # 'concat' | 'mean' | 'max'
        'shared_encoder': False,  # Share weights across planes
        
        # Comparison baseline model from: https://github.com/ChiariRay/AnkleNet.git
        'anklenet': {
            'base_channels': 16,
            'dim': 128,
            'depth': 2,
            'heads': 4,
            'dropout': 0.2,
        },
        
        # Comparison baseline model from: https://github.com/MisaOgura/MRNet.git
        'mrnet': {
            'base_channels': 16,
            'dropout': 0.5,
        },
        
        # Comparison baseline model from: https://github.com/zqiuak/CoPAS
        'copas': {
            'depth': 18,              # ResNet depth
            'use_co_attention': True, # Enable co-plane attention
            'dropout': 0.05,
        },
        
    },
    
    # ==================== Training Configuration ====================
    'training': {
        'epochs': 100,
        'batch_size': 20,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'data_augment': 'aggressive',  # 'basic' | 'light' | 'none'
        'use_balanced_sampling': False,  # Balance positive/negative samples
        'loss_function': 'bce',
        
        # Per-label configuration (optional, for fine-tuning)        
        # Loss function: 'bce' | 'bce_weighted' | 'focal' | 'asymmetric'
        'label_specific': {
            'ACL': {'loss_function': 'bce', },       
            'PCL': {'loss_function': 'focal', 'alpha': 0.5, 'gamma': 2.0},     
            'MCL': {'loss_function': 'bce', },      
            'LCL': {'loss_function': 'asymmetric', 'gamma_neg': 2, 'gamma_pos': 2, 'clip': 0.05,},
        },
    },
    
    # ==================== Visualization Configuration ====================
    'visualization': {
        'enable': False, 
        'roc_curves': True,
        'confusion_matrices': False,
        'prediction_distribution': False,
    }
}
