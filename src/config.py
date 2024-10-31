import torch

class Config:
    # Data settings
    NUM_EDGE_TYPES = 3
    SAMPLE_SIZES = [25, 10, 5]
    DATA_ROOT = 'data/'
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Vulnerability categories
    VULNERABILITY_CATEGORIES = [
        'access_control',
        'arithmetic',
        'denial_of_service',
        'reentrancy',
        'unchecked_low_level_calls'
    ]
    NUM_CLASSES = len(VULNERABILITY_CATEGORIES)
    
    # Model settings
    GNN_INPUT_DIM = 32
    GNN_HIDDEN_DIM = 64
    GNN_OUTPUT_DIM = 128
    TRANSFORMER_MODEL_NAME = 'microsoft/codebert-base'
    
    # Training settings
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Loss function settings
    LOSS_TYPE = 'BCE'  # Binary Cross Entropy for multi-label
    CLASS_WEIGHTS = None  # Can be set based on class distribution
    
    # Paths
    MODEL_SAVE_PATH = 'models/'
    LOG_PATH = 'logs/'
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15