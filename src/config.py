# config.py

class Config:
    # Data settings
    DATA_ROOT = 'data/'
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Model settings
    GNN_INPUT_DIM = 32
    GNN_HIDDEN_DIM = 64
    GNN_OUTPUT_DIM = 128
    TRANSFORMER_MODEL_NAME = 'microsoft/codebert-base'
    
    # Training settings
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    MODEL_SAVE_PATH = 'models/'
    LOG_PATH = 'logs/'