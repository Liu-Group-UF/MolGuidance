import os
import argparse
import json
import random
from pathlib import Path
from time import time
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import dgl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from molguidance.data_processing.data_module import MoleculeDataModule
from molguidance.model_utils.load import data_module_from_config, model_from_config, read_config_file
from molguidance.property_regressor.gvp_regressor import GVPRegressor
from molguidance.models.flowmol import FlowMol
from molguidance.data_processing.utils import get_batch_idxs, get_upper_edge_mask

class TimeAwareGVPRegressor(GVPRegressor):
    """
    Extension of GVPRegressor that works with intermediate structures in the flow matching process.
    Uses the current features (at time t) instead of ground truth final features.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        n_discrete_features = 11 # Number of discrete features: atom types=5， charge_type=6
        self.scalar_embedding = nn.Sequential(
            nn.Linear(n_discrete_features, self.scalar_size),
            nn.SiLU(),
            nn.Linear(self.scalar_size, self.scalar_size),
            nn.SiLU(),
            nn.LayerNorm(self.scalar_size)
        )

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Forward pass that uses the current state features (_t) if available, 
        falling back to ground truth features (_1_true) if not."""
        
        # First check if we have intermediate features '_t'
        # If not, fall back to ground truth '_1_true' features
        # Check if we have intermediate features '_t'
        # has_intermediate = 'a_t' in g.ndata and 'x_t' in g.ndata and 'e_t' in g.edata
        has_intermediate = 'a_t' in g.ndata and 'c_t' in g.ndata and 'x_t' in g.ndata and 'e_t' in g.edata

        if has_intermediate:
            # Process intermediate features with mask dimension
            atom_types = g.ndata['a_t'].float()
            charge_features = g.ndata['c_t'].float()  
            positions = g.ndata['x_t']
            edge_features = g.edata['e_t']
            
            # Handle dimension differences:
            # 1. For atom types and edge features: Remove the last dimension (mask token)
            # a_t is [batch_size, n_atom_types + 1], but we need [batch_size, n_atom_types]
            atom_types = atom_types[:, :-1]  # Remove the mask dimension
            charge_features = charge_features[:, :-1]  # Remove the mask dimension
            edge_features = edge_features[:, :-1]  # Remove the mask dimension
            
            # 2. For continuous features (positions), no dimension adjustment is needed
            # x_t is already [batch_size, 3] same as x_1_true
        else:
            # Use ground truth features
            atom_types = g.ndata['a_1_true'].float()
            charge_features = g.ndata['c_1_true'].float()  
            positions = g.ndata['x_1_true']
            edge_features = g.edata['e_1_true']
        
        # Continue with the standard GVPRegressor forward logic
        # Embed initial features
        # scalar_feats = self.scalar_embedding(atom_types)
        scalar_feats = self.scalar_embedding(torch.cat([atom_types, charge_features], dim=-1)) 
        edge_feats = self.edge_embedding(edge_features)
        
        # Rest of the forward method remains the same...
        # Initialize vector features
        num_nodes = g.num_nodes()
        vector_feats = torch.zeros((num_nodes, self.vector_size, 3), device=g.device)
        
        # Process through GVP layers with optional position and edge updates
        x_diff, d = self.precompute_distances(g, positions)
        
        for conv_idx, conv in enumerate(self.conv_layers):
            # Perform convolution
            scalar_feats, vector_feats = conv(
                g,
                scalar_feats=scalar_feats,
                coord_feats=positions,
                vec_feats=vector_feats,
                edge_feats=edge_feats,
                x_diff=x_diff,
                d=d
            )
            
            # Update positions and edge features if enabled
            if self.update_positions and conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:
                updater_idx = conv_idx // self.convs_per_update
                positions = self.node_position_updaters[updater_idx](scalar_feats, positions, vector_feats)
                x_diff, d = self.precompute_distances(g, positions)
                edge_feats = self.edge_updaters[updater_idx](g, scalar_feats, edge_feats, d=d)
            
        # Store final node representations
        g.ndata['h'] = scalar_feats
        
        # Perform graph pooling based on specified method
        if self.pooling_type == 'attention':
            # Compute attention weights
            attention_weights = torch.sigmoid(self.attention(scalar_feats))
            g.ndata['attn'] = attention_weights
            pooled = dgl.readout_nodes(g, 'h', weight='attn')
        elif self.pooling_type == 'sum':
            pooled = dgl.readout_nodes(g, 'h', op='sum')
        else:  # default to mean pooling
            pooled = dgl.readout_nodes(g, 'h', op='mean')
        
        # Predict properties
        predictions = self.graph_predictor(pooled)
        
        return predictions

class PropertyPredictorModule(pl.LightningModule):
    """
    PyTorch Lightning module for training a property predictor that works at different stages
    of the flow matching trajectory. This will be used for classifier guidance in molecule generation.
    """
    
    def __init__(
        self,
        model_config,
        learning_rate=5e-4,
        weight_decay=1e-4,
        scheduler_patience=10,
        scheduler_factor=0.5,
        loss_type="l1",
        flowmol_config=None,  # Configuration for initializing FlowMol 
        time_conditioning=True,
        sigma_noised=1.0,  # Add this parameter
        eps=1e-9,          # Add this parameter
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize property predictor model based on TimeAwareGVPRegressor
        self.model = TimeAwareGVPRegressor(**model_config)
        
        # Whether to condition on time
        self.time_conditioning = time_conditioning
        
        # Add time embedding if we're conditioning on time
        if self.time_conditioning:
            self.time_embedding_dim = 256
            self.time_embedding = nn.Sequential(
                nn.Linear(1, self.time_embedding_dim),
                nn.SiLU(),
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
                nn.SiLU(),
                nn.LayerNorm(self.time_embedding_dim)
            )
        
        # Optional FlowMol configuration
        self.flowmol_config = flowmol_config
        self.flowmol = None
        
        # Loss function
        if loss_type.lower() == "mse":
            self.loss_fn = F.mse_loss
        elif loss_type.lower() == "l1":
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Add sigma parameters for Gaussian log probability computation
        self.sigma_noised = sigma_noised
        self.eps = eps

        # Learnable parameter for sigma at t=1 (clean data)
        self.log_sigma_unnoised = nn.Parameter(torch.log(torch.tensor(sigma_noised)))

    @property
    def sigma_unnoised(self) -> torch.Tensor:
        """Return the unnoised sigma based on the model parameter."""
        return torch.exp(self.log_sigma_unnoised)

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return sigma(t) as a linear interpolation between:
        - Noised sigma at t=0 (fully masked/noisy state)  
        - Unnoised sigma at t=1 (clean/final state)
        
        Args:
            t: Time tensor of shape (batch_size,)
            
        Returns:
            sigma(t) tensor of shape (batch_size,)
        """
        # At t=1: clean data (lower uncertainty) -> sigma_unnoised
        # At t=0: noisy data (higher uncertainty) -> sigma_noised
        return t * self.sigma_unnoised + (1 - t) * self.sigma_noised
    
    def log_prob(self, y_target: torch.Tensor, g: dgl.DGLGraph, t: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(y_target | molecular_state, t) using Gaussian assumption.
        
        Args:
            y_target: Target property values, shape (batch_size,)
            g: DGL graph representing molecular state at time t
            t: Time tensor, shape (batch_size,)
            
        Returns:
            Log probabilities, shape (batch_size,)
        """
        # Get prediction from the model
        y_pred = self.forward(g, t).squeeze()  # Shape: (batch_size,)
        
        # Ensure y_target is on the same device and has correct shape
        y_target = y_target.to(y_pred.device).squeeze()
        
        # Get time-dependent sigma
        sigma_t = self.get_sigma_t(t).squeeze() + self.eps  # Add small epsilon for numerical stability
        log_sigma_t = torch.log(sigma_t)
        
        # Compute Gaussian log probability: log N(y_target; y_pred, sigma_t^2)
        square_diff = (y_target - y_pred) ** 2 / (2 * sigma_t ** 2)
        log_prob = -square_diff - log_sigma_t - 0.5 * math.log(2 * math.pi)
        
        return log_prob

    def setup(self, stage=None):
        """
        Initialize a new FlowMol instance if needed for interpolation.
        """
        if self.flowmol is None and self.flowmol_config is not None:
            try:
                # Initialize a new FlowMol instance using the provided configuration
                self.flowmol: FlowMol = model_from_config(self.flowmol_config)
                self.flowmol.to(self.device)
                print("Successfully initialized FlowMol for interpolation")
            except Exception as e:
                print(f"Warning: Could not initialize FlowMol: {e}")
                print("Will fall back to training on real data only")

    def forward(self, g, t=None):
        """
        Forward pass through the property predictor.
        
        Args:
            g: A DGLGraph representing the molecule
            t: Optional timestep tensor (batch_size,) for conditioning
            
        Returns:
            Property prediction
        """
        # If timestep is provided and we're conditioning on time, add it as a feature
        if t is not None and self.time_conditioning:
            # Ensure t is the right shape (batch_size, 1)
            if t.dim() == 1:
                t = t.unsqueeze(1)
            
            # Embed timestep
            t_emb = self.time_embedding(t)
            
            # Add as graph-level feature
            g.graph_emb = t_emb
        
        # Forward pass through the GVP regressor
        return self.model(g)

    def sample_at_timepoint(self, g, t):
        """
        Sample a graph at a specific timepoint using the vector field interpolation.
        
        Args:
            g: Original graph
            t: Timestep tensor (batch_size,)
            
        Returns:
            Interpolated graph at timepoint t
        """
        assert self.flowmol is not None, "FlowMol is not initialized. Cannot sample at timepoint."
        
        # Get batch indices and edge mask
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)
        upper_edge_mask = get_upper_edge_mask(g)
        
        # Sample conditional path using vector field interpolation
        with torch.no_grad():
            # Make a clone to avoid modifying the original graph
            g_clone = g.clone()
            
            # Use the vector field to sample at the specified timepoint
            g_t = self.flowmol.vector_field.sample_conditional_path(
                g_clone, t, node_batch_idx, edge_batch_idx, upper_edge_mask
            )
        
        return g_t

    def on_save_checkpoint(self, checkpoint):
        # Store flowmol and flowmol_config temporarily
        temp_flowmol = self.flowmol
        temp_flowmol_config = self.flowmol_config
        
        # Set both to None
        self.flowmol = None
        self.flowmol_config = None
        
        # Let PyTorch Lightning do its normal saving
        checkpoint = super().on_save_checkpoint(checkpoint)
        
        # Restore flowmol and flowmol_config
        self.flowmol = temp_flowmol
        self.flowmol_config = temp_flowmol_config
        
        return checkpoint

    def state_dict(self):
        """Override to filter out flowmol parameters."""
        state_dict = super().state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('flowmol.')}
        return filtered_state_dict

    def load_state_dict(self, state_dict, strict=False):
        """Override to handle loading without flowmol parameters."""
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('flowmol.')}
        return super().load_state_dict(filtered_state_dict, strict=False)

    def compute_loss(self, g, target, t_fix=None):
        """
        Compute loss for the property predictor at a given timepoint or random timepoints.
        
        Args:
            g: Input graph
            target: Target property value
            t_fix: Fixed timepoint (if None, random timepoints are sampled)
            
        Returns:
            Tuple of (loss, absolute error)
        """
        device = g.device
        batch_size = g.batch_size
        
        # Ensure target is on the same device as the graph
        target = target.to(device)

        # Sample timepoint(s)
        if t_fix is None:
            # Random timepoints for each molecule in batch
            t = torch.rand(batch_size, device=device)
        else:
            # Fixed timepoint for all molecules
            t = torch.ones(batch_size, device=device) * t_fix
        
        # Get interpolated molecule at timepoint t
        g_t = self.sample_at_timepoint(g, t)
        
        # Predict property
        pred = self(g_t, t).squeeze()
        
        # Compute loss
        loss = self.loss_fn(pred, target)
        
        # Compute absolute error for logging
        abs_error = (pred - target).abs().detach()
        
        return loss, abs_error

    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: The batch (a DGLGraph)
            batch_idx: Batch index
            
        Returns:
            Loss
        """
        g = batch  # The batch is already a DGLGraph in your dataloader
        target = g.prop
        batch_size = g.batch_size
        
        # Compute loss with random timepoints
        loss, abs_error = self.compute_loss(g, target)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_mae", abs_error.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: The batch (a DGLGraph)
            batch_idx: Batch index
            
        Returns:
            Loss
        """
        g = batch  # The batch is already a DGLGraph in your dataloader
        target = g.prop
        batch_size = g.batch_size
        
        # # If we have a FlowMol model, evaluate at multiple timepoints
        # if self.flowmol is not None:
        #     # Test at multiple fixed timepoints
        #     all_losses = []
        #     all_errors = []
            
        #     # Evaluate at 10 fixed timepoints
        #     for t_val in torch.linspace(0, 1, 11, device=g.device):
        #         loss, abs_error = self.compute_loss(g, target, t_fix=t_val)
        #         all_losses.append(loss)
        #         all_errors.append(abs_error.mean())
                
        #         # Log loss at each timepoint
        #         self.log(f"val_loss_t{t_val:.1f}", loss, on_epoch=True, batch_size=batch_size)
        #         self.log(f"val_mae_t{t_val:.1f}", abs_error.mean(), on_epoch=True, batch_size=batch_size)
            
        #     # Average loss over all timepoints
        #     avg_loss = torch.stack(all_losses).mean()
        #     avg_mae = torch.stack(all_errors).mean()
            
        #     self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        #     self.log("val_mae", avg_mae, on_epoch=True, prog_bar=True, batch_size=batch_size)
            
        #     return avg_loss
        # else:
        #     # If no FlowMol model, just evaluate on clean data
        loss, abs_error = self.compute_loss(g, target)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_mae", abs_error.mean(), on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
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
            }
        }


def train_property_predictor(config_path, flowmol_config_path=None, checkpoint_path=None):
    """
    Training function for property predictor.
    
    Args:
        config_path: Path to GVPRegressor configuration file
        flowmol_config_path: Path to FlowMol configuration file (optional)
        checkpoint_path: Path to resume training from a checkpoint
        
    Returns:
        Tuple of (trained model, trainer)
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get property name and create output directory
    property_name = config["dataset"]["conditioning"]["property"]
    output_dir = Path(config["training"]["output_dir"]) / f"property_predictor_{property_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Create data module
    data_module = data_module_from_config(config)
    flowmol_config = read_config_file(flowmol_config_path)
    
    # Create model
    if checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = PropertyPredictorModule.load_from_checkpoint(checkpoint_path)
        
        # Update FlowMol configuration if provided
        if flowmol_config is not None:
            model.flowmol_config = read_config_file(flowmol_config_path)
            model.flowmol = None  # Force reinitialization in setup()
    else:
        model = PropertyPredictorModule(
            model_config=config["model"],
            learning_rate=config["training"].get("learning_rate", 1e-3),
            weight_decay=config["training"].get("weight_decay", 1e-4),
            scheduler_patience=config["training"].get("scheduler_patience", 10),
            scheduler_factor=config["training"].get("scheduler_factor", 0.5),
            loss_type=config["training"].get("loss_type", "l1"),
            flowmol_config=flowmol_config,
            time_conditioning=config["training"].get("time_conditioning", True),
        )
    
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Create trainer
    trainer_config = config["training"]["trainer_args"]
    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=trainer_config.get("devices", "auto"),
        max_epochs=trainer_config.get("max_epochs", 100),
        gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
        log_every_n_steps=10,
        deterministic=False,
    )
    
    # Train model
    trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    
    return model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train property predictor for classifier guidance")
    parser.add_argument("--config", type=str, required=True, help="Path to GVPRegressor configuration file")
    parser.add_argument("--flowmol_config", type=str, help="Path to FlowMol configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to resume training from checkpoint")
    args = parser.parse_args()
    
    
    # Train model
    model, trainer = train_property_predictor(
        args.config,
        args.flowmol_config,
        args.checkpoint
    )