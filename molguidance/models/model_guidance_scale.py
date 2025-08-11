from typing import Dict, List, Union, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import dgl
from molguidance.models.flowmol import FlowMol
from molguidance.data_processing.utils import get_batch_idxs, get_upper_edge_mask, build_edge_idxs, get_edge_batch_idxs
from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.models.ctmc_vector_field import CTMCVectorField, PROPERTY_MAP
from molguidance.models.classifier_free_guidance import ZerosEmbedding
import copy
import numpy as np
import random


class ModelGuidance(FlowMol):
    """
    Extended FlowMol model with Model Guidance capabilities, based on the paper
    "Diffusion Models without Classifier-free Guidance".
    
    During training, it directly learns the guided distribution by modifying the output
    with the guidance term. 
    
    During sampling, only conditional generation is used with a single forward pass,
    eliminating the need for the two forward passes required by CFG.
    """
    def __init__(self, *args, 
                 min_scale: float = 0.0,        # Minimum scale for guidance
                 max_scale: float = 1.0,        # Maximum scale for guidance
                 drop_ratio: float = 0.1,      # Ratio for dropping condition to train unconditional model
                 training_mode: bool = True,   # Whether in training mode
                 mg_high:float = 0.75,
                 data_ratio: float = 0.2,
                 mg_start_step: int = 10_000,  # training step to start using MG loss
                 use_scale_aware: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.drop_ratio = drop_ratio
        self.training_mode = training_mode
        self.mg_high = mg_high
        self.data_ratio = data_ratio
        self.mg_start_step = mg_start_step
        self.use_scale_aware = use_scale_aware

        vector_field_config = kwargs.get("vector_field_config", {})

        # Create unconditional embedder (zero embedding)
        self.unconditional_embedder = ZerosEmbedding(hidden_dim=self.property_embedding_dim)

        if use_scale_aware:
            self.vector_field = ScaleAwareMGVectorField(
                n_atom_types=self.n_atom_types,
                canonical_feat_order=self.canonical_feat_order,
                interpolant_scheduler=self.interpolant_scheduler, 
                n_charges=self.n_atom_charges, 
                n_bond_types=self.n_bond_types,
                exclude_charges=self.exclude_charges,
                property_embedding_dim=self.property_embedding_dim,
                property_embedder=self.property_embedder,
                properties_handle_method=self.properties_handle_method,
                conditional_generation=self.conditional_generation,
                training_mode=training_mode,
                dataset_name=self.dataset_name,
                **vector_field_config
            )

        self.ema_vector_field = copy.deepcopy(self.vector_field)
        for p in self.ema_vector_field.parameters():
            p.requires_grad = False
        self.ema_vector_field.eval()

    @torch.no_grad()
    def update_ema(self, ema_model, model, decay=0.9999):
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):
        # check if self has the attribute batches_per_epoch
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)

        # compute epoch as a float
        epoch_exact = self.current_epoch + batch_idx/self.batches_per_epoch
        self.last_epoch_exact = epoch_exact

        # update the learning rate
        self.lr_scheduler.step_lr(epoch_exact)

        # compute losses
        losses = self(g)

        # create a dictionary of values to log
        train_log_dict = {}
        train_log_dict['epoch_exact'] = epoch_exact

        for key in losses:
            train_log_dict[f'{key}_train_loss'] = losses[key]

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]

        self.log_dict(train_log_dict, sync_dist=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True, sync_dist=True)

        return total_loss

    def on_after_optimizer_step(self):
        # Update EMA after optimization
        ema_decay = 0.9999  # Decay rate for EMA
        self.update_ema(self.ema_vector_field, self.vector_field, decay=ema_decay)   

    def validation_step(self, g: dgl.DGLGraph, batch_idx: int):
        # compute losses
        losses = self(g, is_training=False)

        # create dictionary of values to log
        val_log_dict = {
            'epoch_exact': self.last_epoch_exact
        }

        for key in losses:
            val_log_dict[f'{key}_val_loss'] = losses[key]

        self.log_dict(val_log_dict, batch_size=g.batch_size, sync_dist=True)

        # combine individual losses into a total loss
        total_loss = torch.zeros(1, device=g.device, requires_grad=False)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]

        self.log('val_total_loss', total_loss, prog_bar=True, batch_size=g.batch_size, on_step=True, sync_dist=True)

        return total_loss

    def forward(self, g: dgl.DGLGraph, is_training=True):
        """
        Modified forward pass that implements Model Guidance by modifying the output
        with the guidance term.
        """
        batch_size = g.batch_size
        device = g.device
        
        round = lambda x: int(np.ceil(x)) if (random.random() < x - int(np.floor(x))) else int(np.floor(x))
        num_mg, num_drop = 0, round(batch_size * self.drop_ratio)

        use_mg_loss =  is_training and self.mg_start_step >= 0 and self.global_step >= self.mg_start_step
        if use_mg_loss:
            num_mg = round(batch_size * self.data_ratio) if self.use_scale_aware else (batch_size - num_drop)

        # Sample random guidance values for training
        guidance_values = torch.ones(batch_size, device=device)
        guidance_values[:num_mg] = torch.rand(num_mg, device=device) * (self.max_scale - self.min_scale) + self.min_scale
        guidance_values[num_mg:num_mg+num_drop] = 0.0

        # Check if the attribute loss_fn_dict exists
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=device)    

        # Get batch indices of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # Create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

        # Sample timepoints for each molecule in the batch
        t = torch.rand(batch_size).float().to(device)

        # Make sure prop has correct shape for the property embedder
        g.prop = g.prop.unsqueeze(-1) if g.prop.dim() == 1 else g.prop
        g.prop = g.prop.to(device)
        
        # Move property embedder to the appropriate device
        self.property_embedder = self.property_embedder.to(device)
        self.unconditional_embedder = self.unconditional_embedder.to(device)

        # Construct interpolated molecules
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)

        emb_mix = torch.cat([self.property_embedder(g.prop[:num_mg]),
                             self.unconditional_embedder(g.prop[num_mg:num_mg+num_drop]),
                             self.property_embedder(g.prop[num_mg+num_drop:])], dim=0)
        t_combined = torch.cat([t.unsqueeze(-1), emb_mix], dim=-1).to(device)

        # Forward pass for the conditional vector field
        if self.use_scale_aware:
            vf_output = self.vector_field(g, t_combined, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask, 
                                            guidance_scale=guidance_values)
        else:
            vf_output = self.vector_field(g, t_combined, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)
        

        if use_mg_loss:
            # Get list of unbatched graphs
            unbatched_graphs = dgl.unbatch(g)

            # Create conditional sub-batch
            cond_graphs = unbatched_graphs[:num_mg]
            cond_g = dgl.batch(cond_graphs)        

            with torch.no_grad():
                # Generate conditional and unconditional embeddings
                cond_prop_emb = self.property_embedder(g.prop[:num_mg])
                uncond_prop_emb = self.unconditional_embedder(g.prop[:num_mg])

                # Get batch indices for sub-graph
                cond_node_batch_idx, cond_edge_batch_idx = get_batch_idxs(cond_g)
                cond_upper_edge_mask = get_upper_edge_mask(cond_g)

                # Time tensor for conditional batch
                t_cond = t[:num_mg]
                t_cond_combined = torch.cat([t_cond.unsqueeze(-1), cond_prop_emb], dim=-1).to(device)
                t_uncond_combined = torch.cat([t_cond.unsqueeze(-1), uncond_prop_emb], dim=-1).to(device)

                if self.use_scale_aware:
                    vf_output_cond = self.ema_vector_field(cond_g, t_cond_combined, node_batch_idx=cond_node_batch_idx, 
                                                    upper_edge_mask=cond_upper_edge_mask, guidance_scale=1)
                    vf_output_uncond = self.ema_vector_field(cond_g, t_uncond_combined, node_batch_idx=cond_node_batch_idx,
                                                    upper_edge_mask=cond_upper_edge_mask, guidance_scale=0)
                else:
                    vf_output_cond = self.ema_vector_field(cond_g, t_cond_combined, node_batch_idx=cond_node_batch_idx, 
                                                    upper_edge_mask=cond_upper_edge_mask)
                    vf_output_uncond = self.ema_vector_field(cond_g, t_uncond_combined, node_batch_idx=cond_node_batch_idx,
                                                    upper_edge_mask=cond_upper_edge_mask)

        # get the target (label) for each feature
        targets = {}

        # w_batch = torch.where(t[:num_mg] < self.mg_high, guidance_values[:num_mg] - 1, 0).view(-1, 1)
        
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            if self.parameterization in ['endpoint', 'dirichlet', 'ctmc']:
                target = data_src[f'{feat}_1_true']

            if self.parameterization == 'ctmc' and feat in ['a', 'c', 'e']:
                # First get the one-hot target without argmax
                if feat == "e":
                    target = target[upper_edge_mask]
                
                if use_mg_loss:
                    # Apply guidance to distributions
                    with torch.no_grad():
                        total_nodes = 0
                        for i in range(num_mg):
                            graph = unbatched_graphs[i]
                            num_nodes = graph.num_nodes()
                            
                            if t[i] < self.mg_high:
                                w_val = guidance_values[i] 
                                if w_val > 0:
                                    node_indices = slice(total_nodes, total_nodes + num_nodes)
                                    guidance_diff = vf_output_cond[feat][node_indices] - vf_output_uncond[feat][node_indices]
                                    
                                    # Add guidance and renormalize
                                    target[node_indices] = target[node_indices] + w_val * guidance_diff
                                    target[node_indices] = F.softmax(target[node_indices], dim=-1)
                            
                            total_nodes += num_nodes
                
                # Convert to indices and handle masking
                target = target.argmax(dim=-1)
                
                # Apply masking for CTMC
                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                target[xt_idxs != self.n_cat_dict[feat]] = -100


            if feat == 'x' and use_mg_loss:
                with torch.no_grad():
                    total_nodes = 0                   
                    # Process each graph in the conditional batch
                    for i in range(num_mg):
                        graph = unbatched_graphs[i]
                        num_nodes = graph.num_nodes()
                        
                        # Apply guidance term to nodes in this graph
                        if t[i] < self.mg_high:
                            w_val = guidance_values[i] 
                            if w_val > 0:
                                # Get corresponding node indices in the full batch
                                node_indices = slice(total_nodes, total_nodes + num_nodes)
                                
                                # Get guidance difference from ema model outputs
                                guidance_diff = vf_output_cond[feat][node_indices] - vf_output_uncond[feat][node_indices]
                                # Apply to target
                                target[node_indices] = target[node_indices] + w_val * guidance_diff
                        total_nodes += num_nodes

            targets[feat] = target

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)

        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0

            # compute the losses
            target = targets[feat]
            losses[feat] = self.loss_fn_dict[feat](vf_output[feat], target)*weight

            # Apply reduction if needed
            if self.time_scaled_loss:
                losses[feat] = losses[feat].mean()

        return losses

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, properties_for_sampling:int|float=None, 
        training_mode:bool=True, guidance_scale = 1.5,  # Add guidance scale parameter
        properties_handle_method:str='concatenate_sum', 
        multilple_values_to_one_property: List[float|int] | None = None, 
        dataset_name: str = None,
        **kwargs):   
        """
        Sample molecules with Model-Guidance.
        
        Args:
            n_atoms: Tensor of shape (batch_size,) containing number of atoms per molecule
            n_timesteps: Number of timesteps for integration
            guidance_scale: The guidance scale to use during sampling (default: 1.5)
            
            (Other params same as FlowMol.sample)
        """
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False    

        # Create a batched graph
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)

        g = dgl.batch(g)
        upper_edge_mask = get_upper_edge_mask(g)
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)            

        # Sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)   

        # Create a tensor for the guidance scale that matches the batch size
        # batch_guidance_scale = torch.full((g.batch_size,), guidance_scale, device=device)

        # Setup integration arguments
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'normalization_file_path': normalization_file_path,
            'conditional_generation': conditional_generation,
            'property_name': property_name,
            'properties_for_sampling': properties_for_sampling,
            'training_mode': training_mode,
            'properties_handle_method': properties_handle_method,
            'multilple_values_to_one_property': multilple_values_to_one_property,
            'dataset_name': dataset_name,
        }

        if self.use_scale_aware:
            integrate_kwargs['guidance_scale'] = guidance_scale
        
        if self.parameterization == 'ctmc':
            integrate_kwargs['stochasticity'] = stochasticity
            integrate_kwargs['high_confidence_threshold'] = high_confidence_threshold

        # Now we need to modify the step method in the vector field to pass the guidance scale
        itg_result = self.vector_field.integrate(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames = itg_result
        else:
            g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        if self.parameterization == 'ctmc':
            ctmc_mol = True
        else:
            ctmc_mol = False

        # Build molecule objects
        molecules = []
        for mol_idx, g_i in enumerate(dgl.unbatch(g)):
            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args,
                ctmc_mol=ctmc_mol,
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges))

        return molecules

    def sample_random_sizes(self, n_molecules: int, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, 
        xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, properties_for_sampling:int|float=None, 
        training_mode:bool=True,   
        properties_handle_method:str=None, 
        multilple_values_to_one_property: List[float|int] | None = None,
        number_of_atoms: List[int]|None = None,   
        guide_w=1.5,    
        dataset_name: str = None,
        **kwargs):
        """Sample n_moceules with the number of atoms sampled from the distribution of the training set."""

        if multilple_values_to_one_property is not None and properties_for_sampling is not None:
            raise ValueError('You can not provide both multilple_values_to_one_property and properties_for_sampling, only one of them should be provided')

        # get the number of atoms that will be in each molecules
        if number_of_atoms:
            atoms_per_molecule = torch.tensor(number_of_atoms).to(device)
        else:
            atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        if multilple_values_to_one_property is not None:
            assert len(atoms_per_molecule) == len(multilple_values_to_one_property), \
                f"{len(atoms_per_molecule)} != {len(multilple_values_to_one_property)}"

        return self.sample(atoms_per_molecule, 
            device=device,  
            stochasticity=stochasticity, 
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            normalization_file_path=normalization_file_path,
            conditional_generation=conditional_generation,
            property_name=property_name, # just for sampling process with normalizing 
            properties_for_sampling=properties_for_sampling,
            training_mode=training_mode,
            properties_handle_method=properties_handle_method,
            multilple_values_to_one_property=multilple_values_to_one_property,
            guidance_scale=guide_w,
            dataset_name=dataset_name,
            **kwargs)
        
class ScaleAwareMGVectorField(CTMCVectorField):
    """
    Vector field implementation for Scale-Aware Model Guidance.
    
    This version allows the guidance scale to be an explicit input parameter
    during both training and inference, enabling single-pass sampling with
    adjustable guidance strength.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create scale embedder
        self.scale_embedder = nn.Sequential(
            nn.Linear(1, self.n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(self.n_hidden_scalars, self.n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(self.n_hidden_scalars)
        )

        self.scalar_embedding_cond_updated = nn.Sequential(
            nn.Linear(self.n_hidden_scalars * 2 + self.property_embedding_dim, self.n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(self.n_hidden_scalars, self.n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(self.n_hidden_scalars)
        )
    
    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, 
                guidance_scale: float|torch.Tensor=None,
                apply_softmax=False, remove_com=False):
        """
        Forward pass with integrated guidance scale.
        """
        device = g.device
        
        with g.local_scope():
            # Extract time and property embedding from input
            is_conditional = len(t.shape) > 1 or self.conditional_generation
            
            # Get time tensor
            time_tensor = t[:, 0] if len(t.shape) > 1 else t
            
            # Process scale if provided
            if guidance_scale is not None:
                if isinstance(guidance_scale, (int, float)):
                    # Convert scalar to tensor
                    scale_tensor = torch.ones(g.batch_size, 1, device=device) * guidance_scale
                else:
                    # Use provided tensor
                    scale_tensor = guidance_scale.unsqueeze(-1) if guidance_scale.dim() == 1 else guidance_scale
                    scale_tensor = scale_tensor.to(device)

                # Get scale embedding
                scale_embedding = self.scale_embedder(scale_tensor)
            else:
                # Use zeros if no scale is provided
                scale_embedding = torch.zeros(g.batch_size, self.n_hidden_scalars, device=device)
            
            scale_emb = scale_embedding[node_batch_idx]
            
            base_features = [
                g.ndata['a_t'],
                time_tensor[node_batch_idx].unsqueeze(-1)
            ]
            if not self.exclude_charges:
                base_features.append(g.ndata['c_t'])
            base_features = torch.cat(base_features, dim=-1)
            node_scalar_features = self.scalar_embedding_uncond(base_features)

            try:
                if is_conditional:
                    # Initialize prop_emb as None
                    prop_emb = None

                    # Case 1: Property info in t (for training)
                    if len(t.shape) > 1:
                        prop_emb = t[:, 1:][node_batch_idx]
                    
                    # Case 2: Explicit sampling properties (for sampling)
                    elif self.properties_for_sampling is not None or self.multilple_values_to_one_property is not None:
                        # Convert scalar to tensor properly
                        if self.properties_for_sampling is not None:
                            assert isinstance(self.properties_for_sampling, (int, float))
                        
                        # Load normalization parameters if needed
                        norm_params = None
                        if self.normalization_file_path:
                            norm_params = torch.load(self.normalization_file_path)
                            if self.dataset_name == 'qm9':
                                assert self.property_name is not None, "property_name must be provided when normalization_file_path is set"
                                property_idx = int(PROPERTY_MAP.get(self.property_name))
                                mean = norm_params['mean'][property_idx].item()
                                std = norm_params['std'][property_idx].item()
                            elif self.dataset_name == 'qme14s':
                                mean = norm_params['mean'].item()
                                std = norm_params['std'].item()

                        if self.multilple_values_to_one_property:
                            assert isinstance(self.multilple_values_to_one_property, list)
                            if norm_params is not None:
                                properties_list = [(val - mean) / std for val in self.multilple_values_to_one_property]
                                properties_batch = torch.tensor(properties_list, device=device).view(g.batch_size, 1)
                            else:
                                properties_batch = torch.tensor(self.multilple_values_to_one_property, device=device).view(g.batch_size, 1)
                        else:
                            properties_for_sampling = self.properties_for_sampling
                            if norm_params is not None:
                                properties_for_sampling = (properties_for_sampling - mean) / std
                            properties_batch = torch.full((g.batch_size, 1), properties_for_sampling, device=device)
                            
                        # Get embedding
                        prop_emb = self.property_embedder(properties_batch)
                        
                        # Repeat for each node in graph
                        prop_emb = prop_emb[node_batch_idx]
                    
                    if prop_emb is None:
                        raise ValueError("No property information available for conditional generation")

                    # Handle properties with different methods
                    if self.properties_handle_method == 'concatenate_sum':
                        intermediate_features = torch.cat([node_scalar_features, scale_emb, prop_emb], dim=-1)
                        intermediate_features = self.scalar_embedding_cond_updated(intermediate_features)
                        node_scalar_features = node_scalar_features + intermediate_features
                    elif self.properties_handle_method == 'concatenate':
                        intermediate_features = torch.cat([node_scalar_features, scale_emb, prop_emb], dim=-1)
                        node_scalar_features = self.scalar_embedding_cond_updated(intermediate_features)
                    elif self.properties_handle_method == 'sum':
                        node_scalar_features = node_scalar_features + scale_emb + prop_emb
                    elif self.properties_handle_method == 'multiply':
                        prop_emb = torch.sigmoid(scale_emb + prop_emb) + 0.5 # range (0.5, 1.5)
                        node_scalar_features = node_scalar_features * prop_emb
                    elif self.properties_handle_method == 'concatenate_multiply':
                        intermediate_features = torch.cat([node_scalar_features, scale_emb, prop_emb], dim=-1)
                        intermediate_features = self.scalar_embedding_cond_updated(intermediate_features)
                        intermediate_features = torch.sigmoid(scale_emb + prop_emb) + 0.5 # range (0.5, 1.5)
                        node_scalar_features = node_scalar_features * intermediate_features
                    else:
                        raise ValueError(f"Invalid properties_handle_method: {self.properties_handle_method}")


            except Exception as e:
                print(f"Debug info: is_conditional={is_conditional}, "
                    # f"training_mode={self.training_mode}, "
                    f"t.shape={t.shape}, "
                    f"has_prop={hasattr(g, 'prop')}, "
                    # f"properties_for_sampling={self.properties_for_sampling}"
                    )
                raise e

            # Continue with the standard vector field forward computation
            node_positions = g.ndata['x_t']
            num_nodes = g.num_nodes()
            node_vec_features = torch.zeros((num_nodes, self.n_vec_channels, 3), device=device)
            edge_features = g.edata['e_t']
            edge_features = self.edge_embedding(edge_features)      

            x_diff, d = self.precompute_distances(g)
            for recycle_idx in range(self.n_recycles):
                for conv_idx, conv in enumerate(self.conv_layers):

                    # perform a single convolution which updates node scalar and vector features (but not positions)
                    node_scalar_features, node_vec_features = conv(g, 
                            scalar_feats=node_scalar_features, 
                            coord_feats=node_positions,
                            vec_feats=node_vec_features,
                            edge_feats=edge_features,
                            x_diff=x_diff,
                            d=d
                    )

                    # every convs_per_update convolutions, update the node positions and edge features
                    if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:

                        if self.separate_mol_updaters:
                            updater_idx = conv_idx // self.convs_per_update
                        else:
                            updater_idx = 0

                        node_positions = self.node_position_updaters[updater_idx](node_scalar_features, node_positions, node_vec_features)

                        x_diff, d = self.precompute_distances(g, node_positions)

                        edge_features = self.edge_updaters[updater_idx](g, node_scalar_features, edge_features, d=d)

            
            # predict final charges and atom type logits
            node_scalar_features = self.node_output_head(node_scalar_features)
            atom_type_logits = node_scalar_features[:, :self.n_atom_types]
            if not self.exclude_charges:
                atom_charge_logits = node_scalar_features[:, self.n_atom_types:]

            # predict the final edge logits
            ue_feats = edge_features[upper_edge_mask]
            le_feats = edge_features[~upper_edge_mask]
            edge_logits = self.to_edge_logits(ue_feats + le_feats)

            # project node positions back into zero-COM subspace
            if remove_com:
                g.ndata['x_1_pred'] = node_positions
                g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(g, feat='x_1_pred', op='mean')[node_batch_idx]
                node_positions = g.ndata['x_1_pred']

        # build a dictionary of predicted features
        dst_dict = {
            'x': node_positions,
            'a': atom_type_logits,
            'e': edge_logits
        }
        if not self.exclude_charges:
            dst_dict['c'] = atom_charge_logits

        # apply softmax to categorical features, if requested
        if apply_softmax:
            for feat in dst_dict.keys():
                if feat in ['a', 'c', 'e']: # if this is a categorical feature
                    dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1) # apply softmax to this feature

        return dst_dict

    def integrate(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor, 
        upper_edge_mask: torch.Tensor, n_timesteps: int, 
        visualize=False, 
        dfm_type='campbell',
        stochasticity=8.0, 
        high_confidence_threshold=0.9,
        cat_temp_func=None,
        forward_weight_func=None,
        tspan=None,
        normalization_file_path:str=None,
        conditional_generation:bool=True,
        property_name:str=None,
        properties_for_sampling:int|float=None,
        training_mode:bool=True,
        properties_handle_method:str=None,
        multilple_values_to_one_property: List[float|int] | None = None,
        guidance_scale=None,
        dataset_name: str = None,
        **kwargs):
        """Integrate the trajectories of molecules along the vector field."""
        
        # TODO: this overrides EndpointVectorField.integrate just because it has some extra arguments
        # we should refactor this so that we don't have to copy the entire function
        
        self.properties_for_sampling = properties_for_sampling
        self.property_name = property_name
        self.conditional_generation = conditional_generation
        self.normalization_file_path = normalization_file_path
        self.training_mode = training_mode
        self.properties_handle_method = properties_handle_method
        self.multilple_values_to_one_property = multilple_values_to_one_property

        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func

        # get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # get the timepoint for integration
        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)
        else:
            t = tspan

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']


        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [ init_frame ]
                traj_frames[f'{feat}_1_pred'] = []
    
        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # determine if this is the last integration step
            if s_idx == t.shape[0] - 1:
                last_step = True
            else:
                last_step = False

            # compute next step and set x_t = x_s
            g = self.step(g, s_i, t_i, alpha_t_i, alpha_s_i, 
                alpha_t_prime_i, 
                node_batch_idx, 
                edge_batch_idx, 
                upper_edge_mask, 
                cat_temp_func=cat_temp_func,
                forward_weight_func=forward_weight_func,
                dfm_type=dfm_type,
                stochasticity=stochasticity, 
                high_confidence_threshold=high_confidence_threshold,
                last_step=last_step,
                normalization_file_path=normalization_file_path,
                conditional_generation=conditional_generation,
                property_name=property_name,
                properties_for_sampling=properties_for_sampling, 
                training_mode=training_mode,
                guidance_scale=guidance_scale,
                dataset_name=dataset_name,
                **kwargs)

            if visualize:
                for feat in self.canonical_feat_order:

                    if feat == "e":
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    if feat == 'e':
                        split_sizes = g.batch_num_edges()
                    else:
                        split_sizes = g.batch_num_nodes()
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

                    ep_frame = g_data_src[f'{feat}_1_pred'].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[f'{feat}_1_pred'].append(ep_frame)

        # set x_1 = x_t
        for feat in self.canonical_feat_order:

            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata

            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)


            return g, reshaped_traj_frames
        
        return g

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
             cat_temp_func: Callable,
             forward_weight_func: Callable, 
             dfm_type: str = 'campbell',
             stochasticity: float = 8.0,
             high_confidence_threshold: float = 0.9, 
             last_step: bool = False,
             inv_temp_func: Callable = None,
            normalization_file_path:str=None,
            conditional_generation:bool=True,
            property_name:str=None,
            properties_for_sampling:int|float=None,
            training_mode:bool=True,
            guidance_scale=None,
            dataset_name: str = None,
            ):

        device = g.device

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold

        if dfm_type is None:
            dfm_type = self.dfm_type

        if inv_temp_func is None:
            inv_temp_func = lambda t: 1.0

        if conditional_generation and not training_mode:
            assert self.properties_for_sampling is not None or self.multilple_values_to_one_property is not None , "Properties for sampling must be provided for conditional generation"
        
        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            guidance_scale=guidance_scale,
        )
        
        dt = s_i - t_i

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.ndata['x_t']
        vf = self.vector_field(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])
        g.ndata['x_t'] = x_t + dt*vf*inv_temp_func(t_i)

        # record predicted endpoint for visualization
        g.ndata['x_1_pred'] = x_1.detach().clone()

        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1) # has shape (num_nodes,)

            if feat == 'e':
                xt = xt[upper_edge_mask]

            p_s_1 = dst_dict[feat]
            temperature = cat_temp_func(t_i)
            p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1) # log probabilities

            if dfm_type == 'campbell':


                xt, x_1_sampled = \
                self.campbell_step(p_1_given_t=p_s_1, 
                                xt=xt, 
                                stochasticity=eta, 
                                hc_thresh=hc_thresh, 
                                alpha_t=alpha_t_i[feat_idx], 
                                alpha_t_prime=alpha_t_prime_i[feat_idx],
                                dt=dt, 
                                batch_size=g.batch_size, 
                                batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(), 
                                n_classes=self.n_cat_feats[feat]+1,
                                mask_index=self.mask_idxs[feat],
                                last_step=last_step,
                                batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                                )

            elif dfm_type == 'gat':
                # record predicted endpoint for visualization
                x_1_sampled = torch.cat([p_s_1, torch.zeros_like(p_s_1[:, :1])], dim=-1)

                xt = self.gat_step(
                    p_1_given_t=p_s_1, 
                    xt=xt, 
                    alpha_t=alpha_t_i[feat_idx], 
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    forward_weight=forward_weight_func(t_i),
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                    n_classes=self.n_cat_feats[feat]+1,
                    mask_index=self.mask_idxs[feat],
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                )
                                   
            
            # if we are doing edge features, we need to modify xt and x_1_sampled to have upper and lower edges
            if feat == 'e':
                e_t = torch.zeros_like(g.edata['e_t'])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(g.edata['e_t'])
                e_1_sampled[upper_edge_mask] = x_1_sampled
                e_1_sampled[~upper_edge_mask] = x_1_sampled
                x_1_sampled = e_1_sampled
            
            data_src[f'{feat}_t'] = xt
            data_src[f'{feat}_1_pred'] = x_1_sampled

        return g



