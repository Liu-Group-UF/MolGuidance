from typing import Dict, List, Union, Callable, Optional
import torch
import torch.nn as nn
from torch.distributions import Categorical
import pytorch_lightning as pl
import dgl
import torch.nn.functional as F
from molguidance.models.flowmol import FlowMol
from molguidance.data_processing.utils import get_batch_idxs, get_upper_edge_mask, build_edge_idxs, get_edge_batch_idxs
from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.models.ctmc_vector_field import CTMCVectorField

class ZerosEmbedding(nn.Module):
    """
    Module that returns a tensor of zeros with the specified hidden dimension.
    Used for unconditional generation in classifier-free guidance.
    """
    def __init__(self, hidden_dim: int=256):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor|int, device='cuda') -> torch.Tensor:
        if isinstance(x, int):
            return torch.zeros(x, self.hidden_dim, device=device)
        return torch.zeros(x.size(0), self.hidden_dim, device=x.device)

class SetEmbeddingType:
    """
    Controls whether to use conditional or unconditional embeddings for each
    batch element during training.
    
    Similar to mattergen's SetEmbeddingType, this class creates a mask where
    True indicates using the unconditional embedding and False indicates using
    the conditional embedding.
    """
    def __init__(self, p_unconditional: float = 0.2):
        """
        Args:
            p_unconditional: Probability of using unconditional embedding during training
        """
        self.p_unconditional = p_unconditional
    
    def __call__(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Creates and sets the unconditional embedding mask for the graph.
        
        Args:
            g: DGL graph with property information
            
        Returns:
            DGL graph with _USE_UNCONDITIONAL_EMBEDDING attribute
        """
        # Only proceed if the graph has property information
        if not hasattr(g, 'prop') or g.prop is None:
            return g
        
        batch_size = g.batch_size
        device = g.device
        
        # Generate random mask where True = use unconditional embedding
        # Shape: [batch_size, 1]
        mask = torch.rand(batch_size, 1, device=device) <= self.p_unconditional
        
        # Add the mask to the graph
        g._USE_UNCONDITIONAL_EMBEDDING = mask
        
        return g       

class ClassifierFreeGuidance(FlowMol):
    """
    Extended FlowMol model with Classifier-Free Guidance capabilities.
    
    During training, each batch element randomly uses either conditional
    or unconditional embeddings based on a probability mask.
    
    During sampling, combines conditional and unconditional predictions
    with a guidance scale to control the influence of the conditioning.
    """
    
    def __init__(self, *args, 
                 p_uncond: float = 0.2,  # Probability of training with unconditional embedding
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create SetEmbeddingType controller
        self.embedding_controller = SetEmbeddingType(p_unconditional=p_uncond)

        vector_field_config = kwargs.get("vector_field_config", {}) 

        # Create unconditional embedder (zero embedding)
        self.unconditional_embedder = ZerosEmbedding(hidden_dim=self.property_embedding_dim)
        self.vector_field = CFGVectorField(n_atom_types=self.n_atom_types,
                                            canonical_feat_order=self.canonical_feat_order,
                                            interpolant_scheduler=self.interpolant_scheduler, 
                                            n_charges=self.n_atom_charges, 
                                            n_bond_types=self.n_bond_types,
                                            exclude_charges=self.exclude_charges,
                                            property_embedding_dim=self.property_embedding_dim,
                                            property_embedder=self.property_embedder,
                                            properties_handle_method=self.properties_handle_method,
                                            conditional_generation=self.conditional_generation,
                                            dataset_name=self.dataset_name,
                                            **vector_field_config)
    def forward(self, g: dgl.DGLGraph):
        """
        Modified forward pass that handles both conditional and unconditional embeddings.
        
        For each sample in the batch, we compute both conditional and unconditional
        embeddings, then select between them based on a mask.
        """        
    
        batch_size = g.batch_size
        device = g.device
        
        # Check if the attribute loss_fn_dict exists
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=g.device)    

        # Get batch indices of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # Create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

        # Sample timepoints for each molecule in the batch
        t = torch.rand(batch_size).float().to(device)

        # For training with CFG, generate masks for conditional vs unconditional embeddings
        # Apply embedding controller to generate mask
        g = self.embedding_controller(g)

        # Make sure prop has correct shape
        g.prop = g.prop.unsqueeze(-1) if g.prop.dim() == 1 else g.prop
        g.prop = g.prop.to(device)

        # Move property embedder to the appropriate device
        self.property_embedder = self.property_embedder.to(device)
        self.unconditional_embedder = self.unconditional_embedder.to(device)

        # Get both conditional and unconditional embeddings
        cond_emb = self.property_embedder(g.prop)
        uncond_emb = self.unconditional_embedder(g.prop)

        # Use mask to select between conditional and unconditional embeddings
        # _USE_UNCONDITIONAL_EMBEDDING: True = use unconditional, False = use conditional
        use_uncond = g._USE_UNCONDITIONAL_EMBEDDING.to(device) 
        prop_emb = torch.where(use_uncond, uncond_emb, cond_emb)

        # Combine with time
        t_combined = torch.cat([t.unsqueeze(-1), prop_emb], dim=-1)
        t_combined = t_combined.to(device)

        # construct interpolated molecules
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)

        # forward pass for the vector field
        vf_output = self.vector_field(g, t_combined, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

        # get the target (label) for each feature
        targets = {}
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        assert self.parameterization == "ctmc", "ClassifierFreeGuidance only supports CTMC parameterization"

        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            target = data_src[f'{feat}_1_true']
            if feat == "e":
                target = target[upper_edge_mask]
            if feat in ['a', 'c', 'e']:
                if self.target_blur == 0.0:
                    target = target.argmax(dim=-1)
                else:
                    target = target + torch.randn_like(target)*self.target_blur
                    target = F.softmax(target, dim=-1)

            # for CTMC parameterization, we do not apply loss on already unmasked features
            if feat in ['a', 'c', 'e']:
                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                # note that we use the default ignore_index of the CrossEntropyLoss class here
                target[ xt_idxs != self.n_cat_dict[feat] ] = -100 # set the target to ignore_index when the feature is already unmasked in xt

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

            # when time_scaled_loss is True, we set the reduction to 'none' so that each training example can be scaled by the time-dependent weight.
            # however, this means that we also need to do the reduction ourselves here.
            if self.time_scaled_loss:
                losses[feat] = losses[feat].mean()

        return losses        

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, properties_for_sampling:int|float=None, 
        training_mode:bool=True,   
        properties_handle_method:str='concatenate_sum', 
        multilple_values_to_one_property: List[float|int] | None = None, 
        guide_w={'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0},
        dfm_type='campbell',
        guidance_format: str = "log",  # 'linear' or 'log'
        where_to_apply_guide: str = "probabilities",  # 'probabilities' or 'rate_matrix',
        dataset_name: str = "qm9",
         **kwargs):   
        """
        Sample molecules with classifier-free guidance.
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

        # Setup integration arguments
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'guide_w': guide_w,
            'normalization_file_path': normalization_file_path,
            'conditional_generation': conditional_generation,
            'property_name': property_name,
            'properties_for_sampling': properties_for_sampling,
            'training_mode': training_mode,
            'properties_handle_method': properties_handle_method,
            'multilple_values_to_one_property': multilple_values_to_one_property,
            'dataset_name': dataset_name
        }
        
        if self.parameterization == 'ctmc':
            integrate_kwargs['stochasticity'] = stochasticity
            integrate_kwargs['high_confidence_threshold'] = high_confidence_threshold
            integrate_kwargs['dfm_type'] = dfm_type
            integrate_kwargs['guidance_format'] = guidance_format
            integrate_kwargs['where_to_apply_guide'] = where_to_apply_guide

        # Integrate with guidance
        itg_result = self.vector_field.integrate_with_CFG_guidance(g, node_batch_idx, **integrate_kwargs, **kwargs)

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
        guide_w={'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0}, 
        dfm_type='campbell_rate_matrix',  
        guidance_format: str = "log", 
        where_to_apply_guide: str = "probabilities",
        dataset_name: str = "qm9",
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
            guide_w=guide_w,
            dfm_type=dfm_type,
            guidance_format=guidance_format,
            where_to_apply_guide=where_to_apply_guide,
            dataset_name=dataset_name,
            **kwargs)


class CFGVectorField(CTMCVectorField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unconditional_embedder = ZerosEmbedding(hidden_dim=self.property_embedding_dim)

    def integrate_with_CFG_guidance(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor, 
            upper_edge_mask: torch.Tensor, n_timesteps: int, 
            guide_w={'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0},
            visualize=False,
            dfm_type='campbell_rate_matrix',
            stochasticity=8.0, 
            high_confidence_threshold=0,
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
            guidance_format: str = "linear",
            where_to_apply_guide: str = "probabilities",
            dataset_name: str = "qm9",
            **kwargs):

        self.properties_for_sampling = properties_for_sampling
        self.property_name = property_name
        self.conditional_generation = conditional_generation
        self.normalization_file_path = normalization_file_path
        self.training_mode = training_mode
        self.properties_handle_method = properties_handle_method
        self.multilple_values_to_one_property = multilple_values_to_one_property
        self.dataset_name = dataset_name

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold

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

        # Get alpha values
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # Initialize features
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']

        # Setup visualization if needed
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
                traj_frames[feat] = [init_frame]
                traj_frames[f'{feat}_1_pred'] = []

        # Integration loop
        for s_idx in range(1, t.shape[0]):
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            dt = s_i - t_i
            last_step = (s_idx == t.shape[0] - 1)

            g = self.step_with_CFG_guidance(g, s_i, t_i, 
                                        alpha_t[s_idx - 1], alpha_t[s_idx], alpha_t_prime[s_idx - 1],
                                        node_batch_idx, edge_batch_idx, upper_edge_mask,
                                        guide_w=guide_w,
                                        cat_temp_func=cat_temp_func,
                                        forward_weight_func=forward_weight_func,
                                        dfm_type=dfm_type,
                                        stochasticity=eta,
                                        high_confidence_threshold=hc_thresh,
                                        last_step=last_step,
                                        normalization_file_path=normalization_file_path,
                                        conditional_generation=conditional_generation,
                                        property_name=property_name,
                                        properties_for_sampling=properties_for_sampling,
                                        training_mode=training_mode,
                                        properties_handle_method=properties_handle_method,
                                        multilple_values_to_one_property=multilple_values_to_one_property,
                                        guidance_format=guidance_format,
                                        where_to_apply_guide=where_to_apply_guide,
                                        **kwargs)            

            if visualize:
                for feat in self.canonical_feat_order:
                    if feat == "e":
                        g_data_src = g.edata
                        split_sizes = g.batch_num_edges()
                    else:
                        g_data_src = g.ndata
                        split_sizes = g.batch_num_nodes()

                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

                    ep_frame = g_data_src[f'{feat}_1_pred'].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[f'{feat}_1_pred'].append(ep_frame)

        # Set final values
        for feat in self.canonical_feat_order:
            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata
            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:
            # Reshape trajectory frames
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

    def step_with_CFG_guidance(self, g, s_i, t_i, alpha_t_i, alpha_s_i, alpha_t_prime_i,
                        node_batch_idx, edge_batch_idx, upper_edge_mask,
                        guide_w={'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0},  # Different weights for each feature
                        cat_temp_func=None,
                        forward_weight_func=None,
                        dfm_type='campbell_rate_matrix',
                        stochasticity=8.0,
                        high_confidence_threshold=0.9,
                        last_step=False,
                        normalization_file_path:str=None,
                        conditional_generation:bool=True,
                        property_name:str=None,
                        properties_for_sampling:int|float=None,
                        training_mode:bool=True,
                        properties_handle_method:str=None,
                        multilple_values_to_one_property: List[float|int] | None = None,
                        guidance_format: str = "linear",  # 'linear' or 'log'
                        where_to_apply_guide: str = "probabilities",  # 'probabilities' or 'rate_matrix'
                        **kwargs):

        # Get predictions from unconditional and conditional models

        # Unconditional
        uncond_emb = self.unconditional_embedder(g.batch_size, device=g.device)
        t_batch = torch.full((g.batch_size,), t_i, device=g.device).unsqueeze(-1)
        t_combined_uncond = torch.cat([t_batch, uncond_emb], dim=-1)
        uncond_pred = self(g, t=t_combined_uncond,
                    node_batch_idx=node_batch_idx,
                    upper_edge_mask=upper_edge_mask,
                    apply_softmax=False, # will do softmax later
                    remove_com=True) 
        
        # the forward function in CTMCVectorField class of ctmc_vector_field.py will take care of conditional embeddings,
        # it match case 2 of forward function adn will handle the conditional embeddings with the time t_i
        cond_pred = self(g, t=torch.full((g.batch_size,), t_i, device=g.device),
                                node_batch_idx=node_batch_idx,
                                upper_edge_mask=upper_edge_mask,
                                apply_softmax=False,
                                remove_com=True)

        dt = s_i - t_i

        # Handle positions
        x_1_cond = cond_pred['x']
        x_1_uncond = uncond_pred['x']
        x_t = g.ndata['x_t']
        guide_w_pos = guide_w['x']
        
        vf_cond = self.vector_field(x_t, x_1_cond, alpha_t_i[0], alpha_t_prime_i[0])
        vf_uncond = self.vector_field(x_t, x_1_uncond, alpha_t_i[0], alpha_t_prime_i[0])
        vf = (1 - guide_w_pos) * vf_uncond + guide_w_pos * vf_cond
        
        g.ndata['x_t'] = x_t + dt * vf
        g.ndata['x_1_pred'] = ((1 - guide_w_pos) * x_1_uncond + guide_w_pos * x_1_cond).detach().clone()
       
        # Handle categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue
            
            guide_weight = guide_w[feat] # Here 'feat' will be 'a', 'c', or 'e'

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1)

            if feat == 'e':
                xt = xt[upper_edge_mask]

            # Apply temperature
            temperature = cat_temp_func(t_i)

            if dfm_type == 'campbell':
                # Get log probabilities
                uncond_val = F.softmax(uncond_pred[feat], dim=-1)
                cond_val = F.softmax(cond_pred[feat], dim=-1)
                log_p_uncond = torch.log(uncond_val)
                log_p_cond = torch.log(cond_val)
                # Compute probabilities with guidance
                p_s_1 = torch.exp(log_p_uncond + guide_weight * (log_p_cond - log_p_uncond))

                # Normalize to ensure valid probabilities
                p_s_1 = p_s_1 / p_s_1.sum(dim=-1, keepdim=True)
                p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1)

                xt, x_1_sampled = self.campbell_step(
                    p_1_given_t=p_s_1,
                    xt=xt,
                    stochasticity=stochasticity,
                    hc_thresh=high_confidence_threshold,
                    alpha_t=alpha_t_i[feat_idx],
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                    n_classes=self.n_cat_feats[feat]+1,
                    mask_index=self.mask_idxs[feat],
                    last_step=last_step,
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx
                )

            elif dfm_type == 'campbell_rate_matrix':
                uncond_val = uncond_pred[feat]
                cond_val = cond_pred[feat]

                if where_to_apply_guide == "probabilities":
                    # not sure should we add temperature here or later yet
                    p_1_given_t_uncond = F.softmax(uncond_val, dim=-1)
                    p_1_given_t_cond = F.softmax(cond_val, dim=-1)
                elif where_to_apply_guide == "rate_matrix":
                    p_1_given_t_uncond = F.softmax(uncond_val / temperature, dim=-1)
                    p_1_given_t_cond = F.softmax(cond_val / temperature, dim=-1)

                xt, x_1_sampled = CFGVectorField.campbell_step_with_rate_matrix_cfg(
                    p_1_given_t_uncond=p_1_given_t_uncond,
                    p_1_given_t_cond=p_1_given_t_cond,
                    xt=xt,
                    stochasticity=stochasticity,
                    alpha_t=alpha_t_i[feat_idx],
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    dt=dt,
                    guide_weight=guide_weight,
                    mask_index=self.mask_idxs[feat],
                    n_classes=self.n_cat_feats[feat]+1,
                    uncond_val=uncond_val,
                    cond_val=cond_val,
                    last_step=last_step,
                    eps=1e-9,
                    guidance_format=guidance_format,
                    where_to_apply_guide=where_to_apply_guide,
                    temperature=temperature
                )

            # Handle edge features
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
    
    @staticmethod 
    def campbell_step_with_rate_matrix_cfg(p_1_given_t_uncond: torch.Tensor,
                                        p_1_given_t_cond: torch.Tensor, 
                                        xt: torch.Tensor,
                                        stochasticity: float, 
                                        alpha_t: float, 
                                        alpha_t_prime: float, 
                                        dt: float,
                                        guide_weight: float,
                                        mask_index: int,
                                        n_classes: int,
                                        uncond_val: torch.Tensor,
                                        cond_val: torch.Tensor,
                                        last_step: bool = False,
                                        eps: float = 1e-9,
                                        guidance_format: str = "linear",
                                        where_to_apply_guide: str = "probabilities",
                                        temperature: float = 1.0
                                        ): 
        """
        Modified campbell_step that applies CFG to rate matrices (paper's approach)
        Args:
            p_1_given_t_uncond: Unconditional model predictions probabilities
            p_1_given_t_cond: Conditional model predictions probabilities
            xt: Current state indices [N]
            guide_weight: CFG guidance weight
            uncond_val: Unconditional model's prediction value before applying softmax converted to probabilities
            cond_val: Conditional model's prediction value before applying softmax converted to probabilities
            last_step: Whether this is the last step of integration
            eps: Small value to avoid log(0)
            guidance_format: 'linear' or 'log' for combining rate matrices
            where_to_apply_guide: 'probabilities' or 'rate_matrix' to apply guidance
        """
        device = xt.device
        # print(f"xt shape: {xt.shape}\n")

        if where_to_apply_guide == "rate_matrix":
            # Step 1: Compute separate rate matrices for unconditional and conditional
            R_t_uncond = CFGVectorField._compute_rate_matrix(p_1_given_t_uncond, xt, alpha_t, alpha_t_prime, 
                                                stochasticity, mask_index, n_classes)
            R_t_cond = CFGVectorField._compute_rate_matrix(p_1_given_t_cond, xt, alpha_t, alpha_t_prime,
                                                stochasticity, mask_index, n_classes)

            if guidance_format == "log":      
                # Step 2: Combine rate matrices with guidance
                R_t_guided = torch.exp(
                    (1 - guide_weight) * torch.log(R_t_uncond + eps) + 
                    guide_weight * torch.log(R_t_cond + eps)
                )
            elif guidance_format == "linear":
                R_t_guided = (1 - guide_weight) * R_t_uncond + guide_weight * R_t_cond
            else:
                raise ValueError(f"Invalid guidance_format: {guidance_format}. Choose 'linear' or 'log'.")
        
        elif where_to_apply_guide == "probabilities":
            if guidance_format == "log":
                log_p_uncond = torch.log(p_1_given_t_uncond + eps)
                log_p_cond = torch.log(p_1_given_t_cond + eps)
                p_s_1 = torch.exp((1 - guide_weight) * log_p_uncond + guide_weight * log_p_cond)
            elif guidance_format == "linear":
                p_s_1 = (1 - guide_weight) * p_1_given_t_uncond + guide_weight * p_1_given_t_cond 
            else:
                raise ValueError(f"Invalid guidance_format: {guidance_format}. Choose 'linear' or 'log'.")
            p_s_1 = F.softmax(p_s_1 / temperature, dim=-1) # softmax can do normalization
            R_t_guided = CFGVectorField._compute_rate_matrix(p_s_1, xt, alpha_t, alpha_t_prime,
                                                stochasticity, mask_index, n_classes)

        # Step 3: Re-normalize rate matrix (diagonal = -row_sum)
        R_t_guided.scatter_(-1, xt.unsqueeze(-1), 0.0)  # Clear diagonal
        row_sums = R_t_guided.sum(dim=-1, keepdim=True)
        R_t_guided.scatter_(-1, xt.unsqueeze(-1), -row_sums)  # Set diagonal

        # Step 4: Convert to transition probabilities
        step_probs = (R_t_guided * dt).clamp(min=0.0, max=1.0)

        # Ensure probability conservation
        step_probs.scatter_(-1, xt.unsqueeze(-1), 0.0)
        stay_probs = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, xt.unsqueeze(-1), stay_probs)   
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)     

        # Step 5: Handle last step
        if last_step:
            # For last step, create guided logits with S+1 dimensions
            guided_logits_full = torch.zeros(uncond_val.shape[0], n_classes, device=device)
            guided_logits_full[:, :uncond_val.shape[-1]] = (1 - guide_weight) * uncond_val + guide_weight * cond_val
            guided_logits_full[:, mask_index] = -1e9 # Never choose mask in final step
            
            is_masked = (xt == mask_index)
            xt_new = xt.clone()
            xt_new[is_masked] = guided_logits_full[is_masked].argmax(-1)
        else:
            xt_new = Categorical(step_probs).sample()

        LARGE_NEG = -1e9
        
        log_p_uncond_full = torch.full((p_1_given_t_uncond.shape[0], n_classes), LARGE_NEG, device=device)
        log_p_cond_full = torch.full((p_1_given_t_cond.shape[0], n_classes), LARGE_NEG, device=device)
        
        log_p_uncond_full[:, :p_1_given_t_uncond.shape[-1]] = torch.log(p_1_given_t_uncond + eps)
        log_p_cond_full[:, :p_1_given_t_cond.shape[-1]] = torch.log(p_1_given_t_cond + eps)

        log_p_guided = (1 - guide_weight) * log_p_uncond_full + guide_weight * log_p_cond_full # just for x1 for visualization

        # Debug: Check for NaN before softmax
        if torch.isnan(log_p_guided).any():
            print("Warning: NaN detected in log_p_guided before softmax!")
            print(f"guide_weight: {guide_weight}")
            print(f"log_p_uncond_full stats: min={log_p_uncond_full.min()}, max={log_p_uncond_full.max()}, has_nan={torch.isnan(log_p_uncond_full).any()}")
            print(f"log_p_cond_full stats: min={log_p_cond_full.min()}, max={log_p_cond_full.max()}, has_nan={torch.isnan(log_p_cond_full).any()}")
        
        x1 = Categorical(F.softmax(log_p_guided, dim=-1)).sample() 

        xt_new = F.one_hot(xt_new, num_classes=n_classes).float()
        x1 = F.one_hot(x1, num_classes=n_classes).float()
        
        return xt_new, x1

    @staticmethod # adapted from https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py
    def _compute_rate_matrix(p_1_given_t: torch.Tensor, xt: torch.Tensor,
                            alpha_t: float, alpha_t_prime: float, stochasticity: float,
                            mask_index: int, n_classes: int) -> torch.Tensor:
        """
        Compute rate matrix R_t following the paper's approach
        """
        device = p_1_given_t.device
        N = xt.shape[0] # N.shape = B * D, where B is batch size and D is number of nodes/edges
        S_actual = p_1_given_t.shape[-1]  # Number of actual classes (S)
        
        # Create full rate matrix for all S+1 classes
        R_t = torch.zeros(N, n_classes, device=device)  # Shape: [N, S+1]

        # Masks for current state
        is_masked = (xt == mask_index).float().unsqueeze(-1)  # [N, 1]
        is_unmasked = 1 - is_masked
        
        # Unmasking rates: from mask to any non-mask state
        # Rate = p(x1=j|xt) * (alpha_t_prime + stochasticity*alpha_t) / (1 - alpha_t)
        unmasking_factor = (alpha_t_prime + stochasticity * alpha_t) / (1 - alpha_t + 1e-8)

        # Only fill the actual class positions (0 to S-1), leave mask position (S) as 0
        R_t[:, :S_actual] = is_masked * p_1_given_t * unmasking_factor
        
        # Remasking rates: from actual classes to mask token
        # For nodes that are currently unmasked, add rate to transition to mask
        R_t[:, mask_index] = is_unmasked.squeeze(-1) * stochasticity
        
        return R_t