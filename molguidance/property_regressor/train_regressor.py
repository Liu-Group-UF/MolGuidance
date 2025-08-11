import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import torchmetrics
from typing import Optional, Dict, Any
import yaml
import dgl

from molguidance.data_processing.data_module import MoleculeDataModule
from molguidance.property_regressor.gvp_regressor import GVPRegressor
from molguidance.model_utils.load import data_module_from_config

class GVPRegressorModule(pl.LightningModule):
    """PyTorch Lightning module for training GVP Regressor"""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model with config
        self.model = GVPRegressor(**model_config)
        
        # Initialize metrics
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        
        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, g: dgl.DGLGraph):
        # The input 'g' is a batched DGLGraph from the collate function
        return self.model(g)

    def training_step(self, g: dgl.DGLGraph, batch_idx:int):
        # batch is a batched DGLGraph containing the property values in g.prop
        # Print device info for debugging (only for first batch)
        if batch_idx == 0:
            print(f"Model device: {next(self.parameters()).device}")
            print(f"Graph device: {g.device}")
            print(f"Property device: {g.prop.device}")
        pred = self(g)
        pred = pred.squeeze()
        target = g.prop.to(pred.device)

        # Calculate loss
        loss = self.criterion(pred, target)
        
        # Calculate metrics
        self.train_mae(pred, target)
        self.train_rmse(pred, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=g.batch_size)
        self.log('train_mae', self.train_mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=g.batch_size)
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=g.batch_size)
        
        return loss

    def validation_step(self, g: dgl.DGLGraph, batch_idx:int):
        pred = self(g)
        pred = pred.squeeze()
        target = g.prop.to(pred.device)
        
        # Calculate loss
        loss = self.criterion(pred, target)
        
        # Calculate metrics
        self.val_mae(pred, target)
        self.val_rmse(pred, target)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=g.batch_size)
        self.log('val_mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=g.batch_size)
        self.log('val_rmse', self.val_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=g.batch_size)
        
        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

def train_gvp_regressor(config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
    """
    Main training function for GVP Regressor
    
    Args:
        config_path: Path to the configuration file. If None, uses default config.
        checkpoint_path: Path to a checkpoint to resume training from. If None, starts fresh.
    """
    # Load config
    if config_path is not None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Config path must be provided")

    # Get the property name from config
    property_name = config['dataset']['conditioning']['property']

    # Create output directory
    output_dir = Path(config['training']['output_dir']) / property_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create data module
    data_module: MoleculeDataModule = data_module_from_config(config)

    # Create model or load from checkpoint
    if checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = GVPRegressorModule.load_from_checkpoint(checkpoint_path)
    else:
        model = GVPRegressorModule(
            model_config=config['model'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            scheduler_patience=config['training']['scheduler_patience'],
            scheduler_factor=config['training']['scheduler_factor']
        )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=config['training']['trainer_args']['accelerator'],
        devices=config['training']['trainer_args']['devices'],
        max_epochs=config['training']['trainer_args']['max_epochs'],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True,  # Also save last checkpoint
                filename='gvp-regressor-{epoch:02d}-{val_loss:.4f}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
    )
    
    # Train model
    trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    
    return model, trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GVP Regressor')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint file to resume training')
    args = parser.parse_args()
    
    model, trainer = train_gvp_regressor(args.config, args.resume_from_checkpoint)