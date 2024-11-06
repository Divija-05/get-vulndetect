# src/init.py

import torch
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import Config
from .gnn_module import SmartContractSAGE
from .transformer_module import SmartContractTransformer
from .fusion_layer import EnhancedFusionLayer
from .get_vulndetect_model import GETVulnDetect

def initialize_model(config: Optional[Config] = None) -> GETVulnDetect:
    """Initialize the GET-VulnDetect model with configuration."""
    if config is None:
        config = Config()
        
    model = GETVulnDetect(
        input_dim=config.GNN_INPUT_DIM,
        gnn_hidden_dim=config.GNN_HIDDEN_DIM,
        transformer_hidden_dim=config.GNN_HIDDEN_DIM,
        output_dim=config.NUM_CLASSES,
        gnn_num_layers=3,
        transformer_num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_length=512,
        num_edge_types=config.NUM_EDGE_TYPES,
        use_hierarchical=True,
        gnn_sample_sizes=config.SAMPLE_SIZES
    )
    
    return model

def load_checkpoint(
    model: GETVulnDetect,
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Dict:
    """Load model checkpoint and return state dict."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def save_checkpoint(
    model: GETVulnDetect,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    save_path: Union[str, Path],
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, save_path)
    
    # Save best model if needed
    if is_best:
        best_path = Path(save_path).parent / 'best_model.pth'
        torch.save(checkpoint, best_path)

def setup_training(
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None
) -> tuple:
    """Set up model, optimizer, and device for training."""
    if config is None:
        config = Config()
        
    device = torch.device(config.DEVICE)
    model = initialize_model(config)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.01
    )
    
    start_epoch = 0
    best_metrics = {'f1': 0.0}
    
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(model, checkpoint_path, device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metrics = checkpoint.get('metrics', best_metrics)
        
    return model, optimizer, device, start_epoch, best_metrics

def setup_logging(config: Config) -> None:
    """Set up logging directories and files."""
    log_dir = Path(config.LOG_PATH)
    model_dir = Path(config.MODEL_SAVE_PATH)
    
    # Create directories if they don't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging configuration
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )