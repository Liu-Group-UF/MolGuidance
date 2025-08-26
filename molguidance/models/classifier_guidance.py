from typing import Dict, List, Union, Callable, Optional, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import dgl
import torch.nn.functional as F
from molguidance.models.flowmol import FlowMol
from molguidance.data_processing.utils import get_batch_idxs, get_upper_edge_mask, build_edge_idxs, get_edge_batch_idxs
from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.models.ctmc_vector_field import CTMCVectorField, PROPERTY_MAP
from molguidance.models.classifier_free_guidance import CFGVectorField

class ClassifierGuidanceFlowMol(FlowMol):
    """
    Extension of FlowMol that implements classifier guidance for property-directed generation.
    
    This class uses an external property predictor to guide the sampling process
    toward molecules with desired target properties.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vector_field_config = kwargs.get("vector_field_config", {})
        self.conditional_generation = False
        self.vector_field = ClassifierGuidanceVectorField(n_atom_types=self.n_atom_types,
                                            canonical_feat_order=self.canonical_feat_order,
                                            interpolant_scheduler=self.interpolant_scheduler, 
                                            n_charges=self.n_atom_charges, 
                                            n_bond_types=self.n_bond_types,
                                            exclude_charges=self.exclude_charges,
                                            property_embedding_dim=self.property_embedding_dim,
                                            property_embedder=self.property_embedder,
                                            properties_handle_method=self.properties_handle_method,
                                            conditional_generation=self.conditional_generation,
                                            **vector_field_config)  

    def set_classifier(self, classifier):
        """Set the classifier model for guidance."""
        # Just directly set the classifier on the vector field
        self.vector_field.classifier = classifier
        self.vector_field.classifier.eval()  # Set to evaluation mode  

    def sample(
        self,
        n_atoms: torch.Tensor,
        target_function: Optional[Callable] = None,
        n_timesteps: int = None,
        device: str = "cuda:0",
        feature_guidance_scales: Optional[Dict[str, float]] = {'x': 1.0, 'a': 1.0, 'c': 1.0, 'e': 1.0},
        visualize: bool = False,
        stochasticity: float = None,
        high_confidence_threshold: float = None,
        xt_traj: bool = False,
        ep_traj: bool = False,
        normalization_file_path: str = None,
        conditional_generation: bool = True,
        property_name: str = None,
        properties_for_sampling: Union[int, float] = None,
        training_mode: bool = False,
        properties_handle_method: str = None,
        multilple_values_to_one_property: List[float] = None,
        **kwargs
    ):
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True

        batch_size = n_atoms.shape[0]

        # Get the edge indices for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # Construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)

        # Batch the graphs
        g = dgl.batch(g)

        # Get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # Compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # Sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

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
            'target_function': target_function,
            'feature_guidance_scales': feature_guidance_scales,
        }

        if self.parameterization == 'ctmc':
            integrate_kwargs['stochasticity'] = stochasticity
            integrate_kwargs['high_confidence_threshold'] = high_confidence_threshold

        itg_result = self.vector_field.integrate_with_classifier_guidance(
            g, node_batch_idx, **integrate_kwargs, **kwargs
        )

        if visualize:
            g, traj_frames = itg_result
        else:
            g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        ctmc_mol = (self.parameterization == 'ctmc')

        # Create molecule objects
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

    def sample_random_sizes(
        self,
        n_molecules: int,
        target_function: Optional[Callable] = None,
        device: str = "cuda:0",
        feature_guidance_scales: Optional[Dict[str, float]] = {'x': 1.0, 'a': 1.0, 'c': 1.0, 'e': 1.0},
        stochasticity: float = None,
        high_confidence_threshold: float = None,
        xt_traj: bool = False,
        ep_traj: bool = False,
        normalization_file_path: str = None,
        conditional_generation: bool = True,
        property_name: str = None,
        properties_for_sampling: Union[int, float] = None,
        training_mode: bool = False,
        properties_handle_method: str = None,
        multilple_values_to_one_property: List[float] = None,
        number_of_atoms: List[int] = None,
        **kwargs
    ):
        if multilple_values_to_one_property is not None and properties_for_sampling is not None:
            raise ValueError('You cannot provide both multilple_values_to_one_property and properties_for_sampling')

        # get the number of atoms that will be in each molecules
        if number_of_atoms:
            atoms_per_molecule = torch.tensor(number_of_atoms).to(device)
        else:
            atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        if multilple_values_to_one_property is not None:
            assert len(atoms_per_molecule) == len(multilple_values_to_one_property), \
                f"{len(atoms_per_molecule)} != {len(multilple_values_to_one_property)}"

        return self.sample(
            atoms_per_molecule,
            target_function=target_function,
            device=device,
            feature_guidance_scales=feature_guidance_scales,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            normalization_file_path=normalization_file_path,
            conditional_generation=conditional_generation,
            property_name=property_name,
            properties_for_sampling=properties_for_sampling,
            training_mode=training_mode,
            properties_handle_method=properties_handle_method,
            multilple_values_to_one_property=multilple_values_to_one_property,
            **kwargs
        )


class ClassifierGuidanceVectorField(CTMCVectorField):
    """
    Extension of CTMCVectorField that implements classifier guidance for conditional generation.
    
    This method uses an external property predictor (classifier/regressor) to guide the sampling
    process toward producing molecules with target properties.
    """
    
    def __init__(self, *args, use_taylor_approximation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = None
        self.use_taylor_approximation = use_taylor_approximation

    def integrate_with_classifier_guidance(
        self, 
        g: dgl.DGLGraph, 
        node_batch_idx: torch.Tensor, 
        upper_edge_mask: torch.Tensor, 
        n_timesteps: int,
        target_function: Optional[Callable] = None,
        feature_guidance_scales: Optional[Dict[str, float]] = None,
        visualize: bool = False,
        dfm_type: str = 'campbell',
        stochasticity: float = 8.0,
        high_confidence_threshold: float = 0.9,
        cat_temp_func: Optional[Callable] = None,
        forward_weight_func: Optional[Callable] = None,
        tspan: Optional[torch.Tensor] = None,
        normalization_file_path: Optional[str] = None,
        conditional_generation: bool = True,
        property_name: Optional[str] = None,
        properties_for_sampling: Optional[Union[float, torch.Tensor]] = None,
        training_mode: bool = False,
        properties_handle_method: Optional[str] = None,
        multilple_values_to_one_property: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Integrate trajectories with classifier guidance.
        
        Args:
            g: Input graph
            node_batch_idx: Batch indices for nodes
            upper_edge_mask: Mask for upper triangle of edge matrix
            n_timesteps: Number of integration timesteps
            target_function: Custom target function (optional, if None uses MSE to target_values)
            feature_guidance_scales: Dict of per-feature guidance scales
            visualize: Whether to return trajectory frames for visualization
            dfm_type: Type of diffusion flow matching ('campbell' or 'gat')
            stochasticity: Stochasticity parameter
            high_confidence_threshold: Threshold for high confidence in purity sampling
            cat_temp_func: Function for categorical temperature
            forward_weight_func: Function for forward weight
            tspan: Specific timepoints for integration
            normalization_file_path: Path to normalization file
            conditional_generation: Whether to use conditional generation
            property_name: Name of the property
            properties_for_sampling: Property values for sampling
            training_mode: Whether in training mode
            properties_handle_method: Method to handle properties
            multilple_values_to_one_property: Multiple property values
            
        Returns:
            If visualize=True, returns (g, traj_frames), else returns g
        """    
        assert self.classifier is not None, "Classifier must be set before using classifier guidance"

        # Set default per-feature guidance scales if not provided
        if feature_guidance_scales is None:
            feature_guidance_scales = {
                'x': 1.0,
                'a': 1.0,
                'c': 1.0,
                'e': 1.0
            }    

        # Store properties for sampling
        self.properties_for_sampling = properties_for_sampling
        self.property_name = property_name
        self.conditional_generation = conditional_generation
        self.normalization_file_path = normalization_file_path
        self.training_mode = training_mode
        self.properties_handle_method = properties_handle_method
        self.multilple_values_to_one_property = multilple_values_to_one_property    

        # Set default stochasticity and high confidence threshold if not provided
        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold

        # Set default cat_temp_func and forward_weight_func if not provided
        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func        

        # Get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # Get timepoints for integration
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

        # Define target function if none provided
        if target_function is None:
            # Default target function: minimize MSE to target values
            def default_target_function(g_temp, t):
                """Target function for classifier guidance that works with the time-aware regressor."""
                batch_size = g_temp.batch_size
                device = g_temp.device
                
                # The classifier now naturally uses the '_t' features, so no need to modify anything
                pred = self.classifier(g_temp, t.reshape(-1, 1))

                norm_params = None
                if self.normalization_file_path and self.property_name:
                    norm_params = torch.load(self.normalization_file_path)
                    property_idx = int(PROPERTY_MAP.get(self.property_name))
                    mean = norm_params['mean'][property_idx].item()
                    std = norm_params['std'][property_idx].item()

                # Get the target values
                if self.multilple_values_to_one_property is not None:
                    assert isinstance(self.multilple_values_to_one_property, list)
                    if norm_params is not None:
                        properties_list = [(val - mean) / std for val in self.multilple_values_to_one_property]
                        target_vals = torch.tensor(properties_list, device=device).view(batch_size, 1)
                    else:
                        target_vals = torch.tensor(self.multilple_values_to_one_property, device=device).view(batch_size, 1)
                elif self.properties_for_sampling is not None:
                    # Use the same value for all molecules in the batch
                    properties_for_sampling = self.properties_for_sampling
                    if norm_params is not None:
                        properties_for_sampling = (properties_for_sampling - mean) / std
                    target_vals = torch.full((batch_size, 1), properties_for_sampling, device=device)
                else:
                    # Default to 0 if no target is specified
                    target_vals = torch.zeros_like(pred)
                
                # Compute negative MAE
                result = -torch.abs(pred - target_vals).sum(dim=1)
                # result = -((pred - target_vals) ** 2).sum(dim=1)
                
                return result, target_vals
                
            target_function = default_target_function

        # Integration loop
        for s_idx in range(1, t.shape[0]):
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            dt = s_i - t_i
            last_step = (s_idx == t.shape[0] - 1)

            g = self.step_with_classifier_guidance(
                g=g,
                s_i=s_i,
                t_i=t_i,
                alpha_t_i=alpha_t[s_idx - 1],
                alpha_s_i=alpha_t[s_idx],
                alpha_t_prime_i=alpha_t_prime[s_idx - 1],
                node_batch_idx=node_batch_idx,
                edge_batch_idx=edge_batch_idx, 
                upper_edge_mask=upper_edge_mask,
                target_function=target_function,
                feature_guidance_scales=feature_guidance_scales,
                cat_temp_func=cat_temp_func,
                forward_weight_func=forward_weight_func,
                dfm_type=dfm_type,
                stochasticity=eta,
                high_confidence_threshold=hc_thresh,
                last_step=last_step,
                **kwargs
            )

        # Set final values
        for feat in self.canonical_feat_order:
            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata
            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']
    
        return g

    def step_with_classifier_guidance(
        self, 
        g: dgl.DGLGraph, 
        s_i: torch.Tensor, 
        t_i: torch.Tensor,
        alpha_t_i: torch.Tensor, 
        alpha_s_i: torch.Tensor, 
        alpha_t_prime_i: torch.Tensor,
        node_batch_idx: torch.Tensor, 
        edge_batch_idx: torch.Tensor, 
        upper_edge_mask: torch.Tensor,
        target_function: Callable,
        feature_guidance_scales: Dict[str, float] = None,
        cat_temp_func: Callable = None,
        forward_weight_func: Callable = None,
        dfm_type: str = 'campbell',
        stochasticity: float = 8.0,
        high_confidence_threshold: float = 0.9,
        last_step: bool = False,
        **kwargs
    ):
        """
        Perform a single step of integration with classifier guidance.
        
        Args:
            g: DGLGraph to sample
            s_i: Next timestep
            t_i: Current timestep
            alpha_t_i: Alpha value at current timestep
            alpha_s_i: Alpha value at next timestep
            alpha_t_prime_i: Alpha prime value at current timestep
            node_batch_idx: Batch indices for nodes
            edge_batch_idx: Batch indices for edges
            upper_edge_mask: Mask for upper edges
            target_function: Function to compute target value/energy
            feature_guidance_scales: Dict of per-feature guidance scales
            cat_temp_func: Function to compute temperature for categorical features
            forward_weight_func: Function to compute forward weight
            dfm_type: Type of diffusion flow matching
            stochasticity: Stochasticity parameter
            high_confidence_threshold: Threshold for high confidence
            last_step: Whether this is the last integration step
            
        Returns:
            Updated graph
        """
        device = g.device

        # Calculate timestep difference
        dt = s_i - t_i

        # First do a standard update step
        with torch.no_grad():
            # Neural net prediction for endpoint
            pred_dict = self(
                g, 
                t=torch.full((g.batch_size,), t_i, device=device),
                node_batch_idx=node_batch_idx,
                upper_edge_mask=upper_edge_mask,
                apply_softmax=False,
                remove_com=True,
            )        

        # --- POSITIONS GUIDANCE ---
        # Handle positions (x) with guidance
        x_t = g.ndata['x_t']
        x_1_pred = pred_dict['x']
        
        # Standard position flow
        vf = self.vector_field(x_t, x_1_pred, alpha_t_i[0], alpha_t_prime_i[0])

        # Apply guidance through the target function
        with torch.enable_grad():
            # # Create a temporary graph for the position update with gradient tracking
            g_temp = g.clone()

            g_temp.ndata['x_t'] = x_t.detach().clone().requires_grad_(True)

            # Compute guidance based on target function
            energy, self.y_target = target_function(g_temp, t=t_i)
            energy = energy.sum()
            # Compute gradient of energy with respect to positions
            grad = torch.autograd.grad(energy, g_temp.ndata['x_t'])[0]

            # Normalize gradient
            min_norm = 1.0
            grad_norm = grad.norm(dim=1, keepdim=True)  # [batch_size, 1]

            # Avoid division by zero (or very small numbers)
            eps = 1e-6
            clip_coef = min_norm / (grad_norm + eps)
            clip_coef_clamped = torch.clamp(clip_coef, min=1.0)  # Only scale up, not down since we are in flow matching not diffusion

            grad *= clip_coef_clamped
            
            # # Scale gradient by feature-specific guidance scale
            scale_up_coef = 5 * t_i
            grad = grad * feature_guidance_scales['x'] * scale_up_coef  
            # grad = grad * feature_guidance_scales['x']
            x_next = x_t + dt * vf + grad * dt 

        # Apply modified flow
        g.ndata['x_t'] = x_next
        g.ndata['x_1_pred'] = x_1_pred.detach().clone()
              

        # --- CATEGORICAL GUIDANCE ---
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue  # Already handled positions with your current code
            
            guide_weight = feature_guidance_scales[feat]

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            # Get current categorical state
            xt = data_src[f'{feat}_t'].argmax(-1)
            if feat == 'e':
                xt = xt[upper_edge_mask]

            # Apply temperature
            temperature = cat_temp_func(t_i)

            predict_val = pred_dict[feat]
            p_1_given_t = F.softmax(predict_val / temperature, dim=-1)

            xt, x_1_sampled = self.campbell_step_with_rate_matrix_cg(p_1_given_t= p_1_given_t,
                                                                    xt=xt,
                                                                    stochasticity=stochasticity,
                                                                    alpha_t=alpha_t_i[feat_idx],
                                                                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                                                                    dt=dt,
                                                                    guide_weight=guide_weight,
                                                                    mask_index=self.mask_idxs[feat],
                                                                    n_classes=self.n_cat_feats[feat]+1,
                                                                    feat=feat,
                                                                    g_full=g,
                                                                    upper_edge_mask=upper_edge_mask,
                                                                    last_step=last_step,
                                                                    eps=1e-9)
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

    def campbell_step_with_rate_matrix_cg(self,
                                        p_1_given_t: torch.Tensor,
                                        xt: torch.Tensor,
                                        stochasticity: float, 
                                        alpha_t: float, 
                                        alpha_t_prime: float, 
                                        dt: float,
                                        guide_weight: float,
                                        mask_index: int,
                                        n_classes: int,
                                        feat: str,
                                        g_full: dgl.DGLGraph,
                                        upper_edge_mask: torch.Tensor = None,
                                        last_step: bool = False,
                                        eps: float = 1e-9,
                                        ): 
        device = xt.device
        R_t = CFGVectorField._compute_rate_matrix(p_1_given_t, xt, alpha_t, alpha_t_prime, 
                                                stochasticity, mask_index, n_classes)

        if guide_weight > 0 and self.classifier is not None:
            # alpha_t is the current time
            if self.use_taylor_approximation:
                R_t = self.get_guided_rate_taylor(xt, alpha_t, R_t, guide_weight, mask_index, n_classes,
                                        feat, g_full, upper_edge_mask=upper_edge_mask)
            else:
                R_t = self.get_guided_rate_exact(xt, alpha_t, R_t, guide_weight, mask_index, n_classes,
                                        feat, g_full, upper_edge_mask=upper_edge_mask)
        
        # Set diagonal to negative row sum
        R_t.scatter_(-1, xt.unsqueeze(-1), 0.0)
        row_sums = R_t.sum(dim=-1, keepdim=True)
        R_t.scatter_(-1, xt.unsqueeze(-1), -row_sums)

        # Convert to transition probabilities
        step_probs = (R_t * dt).clamp(min=0.0, max=1.0)
        step_probs.scatter_(-1, xt.unsqueeze(-1), 0.0)
        stay_probs = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, xt.unsqueeze(-1), stay_probs)
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)        

        # Handle last step
        if last_step:
            guided_logits = torch.zeros(xt.shape[0], n_classes, device=device)
            guided_logits[:, :p_1_given_t.shape[-1]] = torch.log(p_1_given_t + eps)
            guided_logits[:, mask_index] = -1e9
            
            is_masked = (xt == mask_index)
            xt_new = xt.clone()
            xt_new[is_masked] = guided_logits[is_masked].argmax(-1)
        else:
            xt_new = torch.distributions.Categorical(step_probs).sample()

        # Create x1 for visualization
        log_p_guided = torch.full((xt.shape[0], n_classes), -1e9, device=device)
        log_p_guided[:, :p_1_given_t.shape[-1]] = torch.log(p_1_given_t + eps)
        x1 = torch.distributions.Categorical(F.softmax(log_p_guided, dim=-1)).sample()

        xt_new = F.one_hot(xt_new, num_classes=n_classes).float()
        x1 = F.one_hot(x1, num_classes=n_classes).float()
        
        return xt_new, x1    
    
    # adapted from https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py
    def get_guided_rate_exact(self, xt: torch.Tensor, t: float, R_t: torch.Tensor, guide_weight: float,
                        mask_index: int, n_classes: int, feat: str, g_full: dgl.DGLGraph, 
                        upper_edge_mask: torch.Tensor = None):
        """
        Apply classifier guidance to the rate matrix.
        
        Args:
            xt: Current categorical state
            t: Current time
            R_t: Rate matrix
            guide_weight: Weight for guidance
            mask_index: Index of the masked category
            n_classes: Number of classes
            
        Returns:
            Guided updated rate matrix
        """
        N = xt.shape[0]
        S = n_classes - 1  # Actual number of classes (excluding mask)
        device = xt.device
        batch_size = g_full.batch_size
        
        # Get current log probability
        with torch.no_grad():
            y_target = self.y_target if hasattr(self, 'y_target') else None
            if y_target is None:
                raise ValueError("y_target must be set before calling get_guided_rate")
            t_batch = torch.full((batch_size,), t, device=device)
            log_prob_current = self.classifier.log_prob(y_target, g_full, t_batch).sum()
        
        # Get all possible transitions
        xt_jumps = ClassifierGuidanceVectorField.get_all_jump_transitions_graph(xt, S, mask_index)  # Shape: (N*S, N)
        
        # Compute log probabilities for all transitions
        log_prob_jumps = []
        
        # Process in smaller batches to avoid memory issues
        batch_size_jumps = min(50, N * S)  # Adjust based on GPU memory
        num_batches = (N * S + batch_size_jumps - 1) // batch_size_jumps
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size_jumps
            end_idx = min((batch_idx + 1) * batch_size_jumps, N * S)
            
            # For this batch of transitions, compute log probabilities
            with torch.no_grad():
                # Clone the graph for each transition
                g_temp = g_full.clone()
                
                # Accumulate log probabilities for this batch
                batch_log_probs = []
                
                for i in range(start_idx, end_idx):
                    # Update the specific feature with the transition
                    xt_temp_indices = xt_jumps[i]
                    xt_temp_onehot = F.one_hot(xt_temp_indices, num_classes=n_classes).float()
                    
                    # Store original feature
                    if feat == 'a':
                        orig_feat = g_temp.ndata['a_t'].clone()
                        g_temp.ndata['a_t'] = xt_temp_onehot
                    elif feat == 'c':
                        orig_feat = g_temp.ndata['c_t'].clone()
                        g_temp.ndata['c_t'] = xt_temp_onehot
                    elif feat == 'e':
                        orig_feat = g_temp.edata['e_t'].clone()
                        # For edges, we need to handle upper and lower edges
                        e_t = torch.zeros_like(g_temp.edata['e_t'])
                        e_t[upper_edge_mask] = xt_temp_onehot
                        e_t[~upper_edge_mask] = xt_temp_onehot
                        g_temp.edata['e_t'] = e_t
                    
                    # Compute log probability for this transition
                    log_prob = self.classifier.log_prob(y_target, g_temp, t_batch).sum()
                    batch_log_probs.append(log_prob)
                    
                    # Restore original feature for next iteration
                    if feat == 'a':
                        g_temp.ndata['a_t'] = orig_feat
                    elif feat == 'c':
                        g_temp.ndata['c_t'] = orig_feat
                    elif feat == 'e':
                        g_temp.edata['e_t'] = orig_feat
                
                log_prob_jumps.extend(batch_log_probs)
        
        # Convert to tensor and reshape
        log_prob_jumps = torch.stack(log_prob_jumps).view(N, S)
        
        # Compute log probability ratios
        log_prob_ratio = log_prob_jumps - log_prob_current
        
        # Scale by guidance weight
        log_prob_ratio = log_prob_ratio * guide_weight
        
        # Clamp to avoid numerical issues
        log_prob_ratio = torch.clamp(log_prob_ratio, max=80.0)
        
        # Compute probability ratios
        prob_ratio = torch.exp(log_prob_ratio)
        
        # Create full probability ratio matrix including mask dimension
        prob_ratio_full = torch.ones(N, n_classes, device=device)
        prob_ratio_full[:, :S] = prob_ratio
        
        # Apply to rate matrix
        R_t_guided = R_t * prob_ratio_full
        
        return R_t_guided
    
    def get_guided_rate_taylor(self, xt: torch.Tensor, t: float, R_t: torch.Tensor, 
                            guide_weight: float, mask_index: int, n_classes: int,
                            feat: str, g_full: dgl.DGLGraph, upper_edge_mask: torch.Tensor = None):
        """
        Efficient Taylor approximation for classifier guidance.
        Instead of evaluating all N*S transitions, we compute gradients once.
        """
        N = xt.shape[0]
        S = n_classes - 1  # Actual number of classes (excluding mask)
        device = xt.device
        batch_size = g_full.batch_size
        
        # One-hot encode current state
        xt_onehot = F.one_hot(xt, num_classes=n_classes).float()  # Shape: (N, n_classes)
        
        # Set up graph for gradient computation
        g_grad = g_full.clone()
        
        # Enable gradients for the specific feature we're updating
        if feat == 'a':
            g_grad.ndata['a_t'] = g_grad.ndata['a_t'].detach().requires_grad_(True)
            feat_tensor = g_grad.ndata['a_t']
        elif feat == 'c':
            g_grad.ndata['c_t'] = g_grad.ndata['c_t'].detach().requires_grad_(True)
            feat_tensor = g_grad.ndata['c_t']
        elif feat == 'e':
            # For edges, only work with upper triangle
            edge_feat = g_grad.edata['e_t'].detach()
            edge_feat_upper = edge_feat[upper_edge_mask].requires_grad_(True)
            # We'll need to map gradients back
            feat_tensor = edge_feat_upper
        
        # Compute log probability and its gradient
        with torch.enable_grad():
            y_target = self.y_target if hasattr(self, 'y_target') else None
            if y_target is None:
                raise ValueError("y_target must be set before calling get_guided_rate_taylor")
            t_batch = torch.full((batch_size,), t, device=device)
            
            # For edge features, we need to handle the mapping
            if feat == 'e':
                # Create a temporary full edge feature tensor
                edge_feat_full = edge_feat.clone()
                edge_feat_full[upper_edge_mask] = feat_tensor
                edge_feat_full[~upper_edge_mask] = feat_tensor  # Symmetric
                g_grad.edata['e_t'] = edge_feat_full
            
            # Compute log probability
            log_prob = self.classifier.log_prob(y_target, g_grad, t_batch).sum()
            
            # Compute gradient with respect to features
            grad_log_prob = torch.autograd.grad(log_prob, feat_tensor, retain_graph=False)[0]
            
            # if feat == 'a':
            #     g_grad.ndata['a_t'] = original_feat
            # elif feat == 'c':
            #     g_grad.ndata['c_t'] = original_feat
            # elif feat == 'e':
            #     g_grad.edata['e_t'] = original_feat

        # grad_log_prob has shape (N, n_classes) - gradient for each node/edge and class
        
        # Taylor approximation: log p(y|x') ≈ log p(y|x) + ∇log p(y|x) · (x' - x)
        # For transitions from current state to other states:
        # log_ratio ≈ ∇log p(y|x)[i,j] - ∇log p(y|x)[i,xt[i]]
        
        # Extract gradient at current state for each node
        grad_at_current = (xt_onehot * grad_log_prob).sum(dim=-1, keepdim=True)  # (N, 1)
        
        # Compute log probability ratios using Taylor approximation
        log_prob_ratio_full = grad_log_prob - grad_at_current  # (N, n_classes)
        
        # We only care about transitions to non-mask states
        log_prob_ratio = log_prob_ratio_full[:, :S]  # (N, S)
        
        # Scale by guidance weight
        log_prob_ratio = log_prob_ratio * guide_weight
        
        # Clamp to avoid numerical issues
        log_prob_ratio = torch.clamp(log_prob_ratio, max=80.0)
        
        # Compute probability ratios
        prob_ratio = torch.exp(log_prob_ratio)
        
        # Create full probability ratio matrix including mask dimension
        prob_ratio_full = torch.ones(N, n_classes, device=device)
        prob_ratio_full[:, :S] = prob_ratio
        
        # Apply to rate matrix
        R_t_guided = R_t * prob_ratio_full
        
        return R_t_guided    

    @staticmethod
    def get_all_jump_transitions_graph(xt: torch.Tensor, S: int, mask_index: int) -> torch.Tensor:
        """
        Gets all possible single-node transitions from current states for graph features.
        
        Args:
            xt: Current state tensor of shape (N,) where N is number of nodes/edges
            S: Number of actual classes (not including mask dimension)
            mask_index: Index of the mask token (typically S)
            
        Returns:
            Tensor of shape (N*S, N) containing all possible single-node transitions
            where each row represents a state that differs from xt in exactly one position
        """
        N = xt.shape[0]
        device = xt.device
        
        # Create N*S copies of input states
        # Shape: (N, 1) -> (N, S) -> (N*S,)
        xt_expand = xt.unsqueeze(1).repeat(1, S).flatten()
        # Shape: (N*S,) -> (N*S, 1) -> (N*S, N)
        xt_expand = xt_expand.unsqueeze(1).repeat(1, N)
        
        # Create indices for which node changes and to what value
        # For each node i, we create S transitions where node i takes values 0, 1, ..., S-1
        node_indices = torch.arange(N, device=device).unsqueeze(1).repeat(1, S).flatten()  # [0,0,...,0, 1,1,...,1, ...]
        new_values = torch.arange(S, device=device).repeat(N)  # [0,1,...,S-1, 0,1,...,S-1, ...]
        
        # Apply transitions
        xt_jumps = xt_expand.clone()
        batch_indices = torch.arange(N*S, device=device)
        xt_jumps[batch_indices, node_indices] = new_values
        
        return xt_jumps