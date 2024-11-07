# src/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .config import Config
from ._init_ import setup_training, save_checkpoint, setup_logging

class VulnerabilityLoss(nn.Module):
    """Custom loss function for vulnerability detection."""
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Binary cross entropy for each vulnerability type
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            weight=self.class_weights
        )
        
        # Focal loss component for handling class imbalance
        gamma = 2.0
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** gamma) * bce_loss
        
        return focal_loss.mean()

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions_list = []
    targets_list = []
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch data
            x = batch.x.to(device)
            edge_index_dict = {
                k: v.to(device) for k, v in batch.edge_index_dict.items()
            }
            code = batch.code
            targets = batch.y.to(device)
            batch_idx = batch.batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(x, edge_index_dict, code, batch_idx)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions_list.append(predictions.detach().cpu())
            targets_list.append(targets.cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate epoch metrics
    predictions = torch.cat(predictions_list)
    targets = torch.cat(targets_list)
    metrics = calculate_metrics(predictions, targets)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, metrics

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    predictions_list = []
    targets_list = []

    for batch in val_loader:
        # Unpack batch data
        x = batch.x.to(device)
        edge_index_dict = {
            k: v.to(device) for k, v in batch.edge_index_dict.items()
        }
        code = batch.code
        targets = batch.y.to(device)
        batch_idx = batch.batch.to(device)
        
        # Forward pass
        predictions = model(x, edge_index_dict, code, batch_idx)
        loss = criterion(predictions, targets)
        
        # Update metrics
        total_loss += loss.item()
        predictions_list.append(predictions.cpu())
        targets_list.append(targets.cpu())
    
    # Calculate metrics
    predictions = torch.cat(predictions_list)
    targets = torch.cat(targets_list)
    metrics = calculate_metrics(predictions, targets)
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, metrics

def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # Convert predictions to binary
    binary_preds = (predictions > threshold).float()
    
    # Calculate precision, recall, F1 per class
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets.numpy(),
        binary_preds.numpy(),
        average='weighted'
    )
    
    # Calculate ROC AUC
    try:
        auc = roc_auc_score(
            targets.numpy(),
            predictions.numpy(),
            average='weighted',
            multi_class='ovr'
        )
    except ValueError:
        auc = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """Main training loop."""
    # Setup
    setup_logging(config)
    model, optimizer, device, start_epoch, best_metrics = setup_training(
        config,
        checkpoint_path
    )
    
    # Initialize loss function
    criterion = VulnerabilityLoss(
        class_weights=torch.tensor(config.CLASS_WEIGHTS).to(device)
        if config.CLASS_WEIGHTS is not None else None
    )
    
    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Evaluate
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        
        # Log metrics
        logging.info(
            f'Epoch {epoch}/{config.NUM_EPOCHS-1} - '
            f'Train Loss: {train_loss:.4f} - '
            f'Val Loss: {val_loss:.4f} - '
            f'Val F1: {val_metrics["f1"]:.4f}'
        )
        
        # Save checkpoint
        is_best = val_metrics['f1'] > best_metrics['f1']
        if is_best:
            best_metrics = val_metrics
            
        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
        save_path = Path(config.MODEL_SAVE_PATH) / checkpoint_name
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            metrics=val_metrics,
            save_path=save_path,
            is_best=is_best
        )
    
    return model

if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    from .data_processing import SmartContractDataset  # Assuming dataset.py contains your SmartContractDataset class
    
    # Load config
    config = Config()
    
    # Create dataset
    dataset = SmartContractDataset(root=config.DATA_PATH)
    
    # Create train and validation dataloaders
    train_data = torch.load(os.path.join(dataset.processed_dir, 'train.pt'))
    val_data = torch.load(os.path.join(dataset.processed_dir, 'val.pt'))
    
    train_loader = DataLoader(
        train_data, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Train model
    model = train(config, train_loader, val_loader)
