# main.py

import argparse
import logging
import torch
from pathlib import Path
from torch_geometric.data import DataLoader
from src.config import Config
from src._init_ import setup_training, save_checkpoint, setup_logging
from src.data_processing import SmartContractDataset
from src.train import train, evaluate

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GET-VulnDetect - Smart Contract Vulnerability Detection")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the model on validation data")
    parser.add_argument('--resume', type=str, default=None, help="Path to a checkpoint to resume training")
    parser.add_argument('--config', type=str, default=None, help="Path to a custom config file (optional)")
    parser.add_argument('--batch_size', type=int, help="Batch size for training and evaluation", default=None)
    parser.add_argument('--epochs', type=int, help="Number of training epochs", default=None)
    return parser.parse_args()

def load_datasets(config):
    """Load datasets and return DataLoader instances."""
    dataset = SmartContractDataset(root=config.DATA_ROOT)
    train_data = torch.load(Path(config.DATA_ROOT) / 'processed' / 'train.pt')
    val_data = torch.load(Path(config.DATA_ROOT) / 'processed' / 'val.pt')

    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader

def main():
    args = parse_arguments()

    # Load configuration
    config = Config()
    if args.config:
        # Load custom configurations here if needed
        pass
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    # Setup logging
    setup_logging(config)

    # Load datasets
    train_loader, val_loader = load_datasets(config)

    if args.train:
        logging.info("Starting training...")
        model = train(config, train_loader, val_loader, checkpoint_path=args.resume)
        logging.info("Training completed successfully.")

    if args.evaluate:
        # Setup model, optimizer, and device (for evaluation only)
        model, _, device, _, _ = setup_training(config, checkpoint_path=args.resume)
        criterion = torch.nn.BCEWithLogitsLoss()

        logging.info("Evaluating the model on validation data...")
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        logging.info(f"Validation Loss: {val_loss:.4f}")
        logging.info(f"Validation Metrics: Precision: {val_metrics['precision']:.4f}, "
                     f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, "
                     f"AUC: {val_metrics['auc']:.4f}")

if __name__ == '__main__':
    main()
