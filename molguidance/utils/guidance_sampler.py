from .propmolflow_util import allowed_properties, load_model_from_ckpt
import math
import torch
import torch.nn.functional as F
import dgl

def sample_molecules_with_guidance(model,
                    n_mols, 
                    property_name,
                    properties_handle_method,
                    multiple_values_to_one_property=None,
                    number_of_atoms = None,
                    guide_w1=1.5,
                    guide_w2=1.5,
                    fixed_property_val = None,
                    normalization_file_path=None,
                    n_timesteps=100,
                    stochasticity=None, 
                    max_batch_size=128, 
                    xt_traj=False, 
                    ep_traj=False,
                    high_confidence_threshold=None,
                    training_mode=False,
                    dataset_name=None,
                    device='cuda'
                    ):
    assert property_name in allowed_properties, f"Property {property_name} is not supported."
    assert (fixed_property_val is None) ^ (multiple_values_to_one_property is None)      

    n_batches = math.ceil(n_mols / max_batch_size)
    molecules = []
    if model.__class__.__name__ == 'ClassifierFreeGuidance':
        guide_w = {}
        guide_w['x'] = guide_w1
        guide_w['a'] = guide_w2
        guide_w['c'] = guide_w2
        guide_w['e'] = guide_w2
    elif model.__class__.__name__ == 'ModelGuidance':
        if not isinstance(guide_w1, (int, float)) or not isinstance(guide_w2, (int, float)):
            raise ValueError(f"guide_w should be a number, but got {type(guide_w1)} and {type(guide_w2)}")
        assert guide_w1 == guide_w2, "guide_w1 and guide_w2 should be equal for ModelGuidance."
        guide_w = guide_w1
    for _ in range(n_batches):
        n_mols_needed = n_mols - len(molecules)
        batch_size = min(n_mols_needed, max_batch_size)
        batch_property = None
        batch_no_atoms = None
        if multiple_values_to_one_property is not None:
            batch_property = multiple_values_to_one_property[len(molecules): len(molecules) + batch_size]
        if number_of_atoms is not None:
            batch_no_atoms = number_of_atoms[len(molecules): len(molecules) + batch_size]
        batch_molecules = model.sample_random_sizes(
            batch_size,
            device=device,
            n_timesteps=n_timesteps,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            properties_for_sampling=None,
            training_mode = training_mode,
            property_name = property_name,
            normalization_file_path = normalization_file_path,
            properties_handle_method = properties_handle_method,
            multilple_values_to_one_property = batch_property,
            number_of_atoms = batch_no_atoms,
            guide_w=guide_w,
            dataset_name=dataset_name,
        )
        molecules.extend(batch_molecules)
    return molecules

def sample_molecules_with_auto_guidance(good_model,
                    bad_model,
                    n_mols, 
                    property_name,
                    properties_handle_method,
                    multiple_values_to_one_property=None,
                    number_of_atoms=None,
                    guide_w=None,
                    fixed_property_val=None,
                    normalization_file_path=None,
                    n_timesteps=100,
                    stochasticity=None, 
                    max_batch_size=128, 
                    xt_traj=False, 
                    ep_traj=False,
                    high_confidence_threshold=None,
                    training_mode=False,
                    device='cuda'):
    assert property_name in allowed_properties, f"Property {property_name} not supported"
    assert (fixed_property_val is None) ^ (multiple_values_to_one_property is None)

    # Load models
    # good_model = load_model_from_ckpt(good_model_ckpt)
    # bad_model = load_model_from_ckpt(bad_model_ckpt)
    # good_model.to(device).eval()
    # bad_model.to(device).eval()

    # Setup guidance weights
    if guide_w is None:
        guide_w = 2.0
    if isinstance(guide_w, (int, float)):
        guide_w = {feat: guide_w for feat in ['x', 'a', 'c', 'e']}
    
    if not fixed_property_val:
        property_value = multiple_values_to_one_property[0]
    else:
        property_value = fixed_property_val

    n_batches = math.ceil(n_mols / max_batch_size)
    molecules = []

    for _ in range(n_batches):
        n_mols_needed = n_mols - len(molecules)
        batch_size = min(n_mols_needed, max_batch_size)
        batch_property = None
        batch_no_atoms = None
        
        if multiple_values_to_one_property is not None:
            batch_property = multiple_values_to_one_property[len(molecules):len(molecules) + batch_size]
        if number_of_atoms is not None:
            batch_no_atoms = number_of_atoms[len(molecules):len(molecules) + batch_size]

        if number_of_atoms is None:
            # Sample random sizes
            atoms_per_molecule = good_model.sample_n_atoms(batch_size).to(device)
        else:
            atoms_per_molecule = torch.tensor(batch_no_atoms, device=device)

        batch_molecules = sample_with_auto_guidance(
            good_model=good_model,
            bad_model=bad_model,
            n_atoms=atoms_per_molecule,
            guide_w=guide_w,
            device=device,
            n_timesteps=n_timesteps,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            properties_for_sampling=property_value,
            property_name=property_name,
            normalization_file_path=normalization_file_path,
            properties_handle_method=properties_handle_method,
            multilple_values_to_one_property=batch_property,
            conditional_generation=True
        )
        molecules.extend(batch_molecules)

    return molecules

def _prepare_property_embedding(model, prop_value, batch_size, device, normalization_file_path=None, property_name=None):
    """Helper function to prepare property embeddings"""
    if normalization_file_path is not None and property_name is not None:
        norm_params = torch.load(normalization_file_path)
        property_idx = PROPERTY_MAP.get(property_name)
        mean = norm_params['mean'][property_idx].item()
        std = norm_params['std'][property_idx].item()
        prop_value = (prop_value - mean) / std
    
    prop_tensor = torch.full((batch_size, 1), prop_value, device=device)
    return model.property_embedder(prop_tensor)

def _handle_categorical_feature(feat, good_model, bad_model, good_pred, bad_pred, guide_w, g, 
                              t_i, alpha_t, alpha_t_prime, s_idx, dt, last_step,
                              node_batch_idx, edge_batch_idx, upper_edge_mask,
                              stochasticity=None, high_confidence_threshold=None):
    """Helper function to handle categorical features (atoms, charges, bonds)"""
    if feat == 'e':
        data_src = g.edata
    else:
        data_src = g.ndata
    
    xt = data_src[f'{feat}_t'].argmax(-1)
    if feat == 'e':
        xt = xt[upper_edge_mask]
    
    # Apply guidance in log space
    log_p_bad = torch.log(bad_pred[feat] + 1e-10)
    log_p_good = torch.log(good_pred[feat] + 1e-10)
    p_s_1 = torch.exp(log_p_bad + guide_w[feat] * (log_p_good - log_p_bad))
    p_s_1 = F.softmax(torch.log(p_s_1)/good_model.vector_field.cat_temp_func(t_i), dim=-1)
    
    # Campbell step
    if hasattr(good_model.vector_field, 'campbell_step'):
        batch_idx = edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx
        xt, x_1_sampled = good_model.vector_field.campbell_step(
            p_1_given_t=p_s_1, xt=xt,
            stochasticity=stochasticity or good_model.vector_field.eta,
            hc_thresh=high_confidence_threshold or good_model.vector_field.hc_thresh,
            alpha_t=alpha_t[s_idx - 1][good_model.canonical_feat_order.index(feat)],
            alpha_t_prime=alpha_t_prime[s_idx - 1][good_model.canonical_feat_order.index(feat)],
            dt=dt, batch_size=g.batch_size,
            batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
            n_classes=good_model.vector_field.n_cat_feats[feat]+1,
            mask_index=good_model.vector_field.mask_idxs[feat],
            last_step=last_step, batch_idx=batch_idx
        )
    
    # Handle edge features specially
    if feat == 'e':
        xt, x_1_sampled = _handle_edge_features(xt, x_1_sampled, g)
    
    return xt, x_1_sampled

from flowmol.data_processing.utils import build_edge_idxs, get_batch_idxs, get_upper_edge_mask
from flowmol.analysis.molecule_builder import SampledMolecule
from flowmol.models.ctmc_vector_field import PROPERTY_MAP
@torch.no_grad()
def sample_with_auto_guidance(good_model, bad_model, n_atoms, guide_w, **kwargs):
    """Sample molecules using auto-guidance with good and bad models"""
    # Initialize graph
    g = _create_batched_graph(n_atoms, kwargs.get('device', 'cuda:0'))
    upper_edge_mask = get_upper_edge_mask(g)
    node_batch_idx, edge_batch_idx = get_batch_idxs(g)
    g = good_model.sample_prior(g, node_batch_idx, upper_edge_mask)
    
    # Setup integration
    t = torch.linspace(0, 1, kwargs.get('n_timesteps', 100), device=kwargs.get('device', 'cuda:0'))
    alpha_t = good_model.interpolant_scheduler.alpha_t(t)
    alpha_t_prime = good_model.interpolant_scheduler.alpha_t_prime(t)
    
    # Initialize features
    _initialize_features(g, good_model)
    
    # Integration loop
    for s_idx in range(1, len(t)):
        # Get predictions from both models
        good_pred, bad_pred = _get_model_predictions(
            good_model, bad_model, g, t, s_idx, 
            kwargs.get('properties_for_sampling'), 
            kwargs.get('normalization_file_path'),
            kwargs.get('property_name')
        )
        
        # Update positions
        _update_positions(g, good_pred, bad_pred, guide_w, t[s_idx] - t[s_idx-1])
        
        # Update categorical features
        for feat in good_model.canonical_feat_order:
            if feat != 'x':
                xt, x_1_sampled = _handle_categorical_feature(
                    feat, good_model, bad_model, good_pred, bad_pred, guide_w, g,
                    t[s_idx-1], alpha_t, alpha_t_prime, s_idx, 
                    t[s_idx] - t[s_idx-1], s_idx == len(t)-1,
                    *get_batch_idxs(g), get_upper_edge_mask(g),
                    kwargs.get('stochasticity'), kwargs.get('high_confidence_threshold')
                )
                if feat == 'e':
                    g.edata[f'{feat}_t'] = xt
                    g.edata[f'{feat}_1_pred'] = x_1_sampled
                else:
                    g.ndata[f'{feat}_t'] = xt
                    g.ndata[f'{feat}_1_pred'] = x_1_sampled
    
    return _build_molecules(g, good_model, kwargs.get('xt_traj', False), kwargs.get('ep_traj', False))

def _create_batched_graph(n_atoms, device):
    """Create a batched graph for molecule generation"""
    edge_idxs_dict = {}
    for n_atoms_i in torch.unique(n_atoms):
        edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

    g = []
    for n_atoms_i in n_atoms:
        edge_idxs = edge_idxs_dict[int(n_atoms_i)]
        g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
        g.append(g_i)

    return dgl.batch(g)

def _initialize_features(g, model):
    """Initialize graph features from prior"""
    for feat in model.canonical_feat_order:
        if feat == 'e':
            g.edata[f'{feat}_t'] = g.edata[f'{feat}_0']
        else:
            g.ndata[f'{feat}_t'] = g.ndata[f'{feat}_0']

def _get_model_predictions(good_model, bad_model, g, t, s_idx, props_sampling, norm_file, prop_name):
    """Get predictions from both models"""
    t_i = t[s_idx - 1]
    batch_size, device = g.batch_size, g.device
    
    # Prepare property embeddings for both models
    prop_emb_good = _prepare_property_embedding(good_model, props_sampling, batch_size, device, norm_file, prop_name)
    prop_emb_bad = _prepare_property_embedding(bad_model, props_sampling, batch_size, device, norm_file, prop_name)
    
    # Combine time and property embeddings
    t_tensor = torch.full((batch_size, 1), t_i, device=device)
    t_combined_good = torch.cat([t_tensor, prop_emb_good], dim=-1)
    t_combined_bad = torch.cat([t_tensor, prop_emb_bad], dim=-1)
    
    node_batch_idx, edge_batch_idx = get_batch_idxs(g)
    upper_edge_mask = get_upper_edge_mask(g)
    
    # Get predictions from both models
    good_pred = good_model.vector_field(g, t_combined_good, node_batch_idx, upper_edge_mask, apply_softmax=True, remove_com=True)
    bad_pred = bad_model.vector_field(g, t_combined_bad, node_batch_idx, upper_edge_mask, apply_softmax=True, remove_com=True)
    
    return good_pred, bad_pred

def _update_positions(g, good_pred, bad_pred, guide_w, dt):
    """Update position features using guided predictions"""
    x_t = g.ndata['x_t']
    guide_w_pos = guide_w['x']
    
    # Apply guidance
    vf = bad_pred['x'] - x_t + guide_w_pos * (good_pred['x'] - bad_pred['x'])
    
    # Update positions
    g.ndata['x_t'] = x_t + dt * vf
    g.ndata['x_1_pred'] = bad_pred['x'] + guide_w_pos * (good_pred['x'] - bad_pred['x'])

def _handle_edge_features(xt, x_1_sampled, g):
    """Handle edge feature updates"""
    e_t = torch.zeros_like(g.edata['e_t'])
    e_1_sampled = torch.zeros_like(g.edata['e_t'])
    
    upper_edge_mask = get_upper_edge_mask(g)
    e_t[upper_edge_mask] = xt
    e_t[~upper_edge_mask] = xt
    e_1_sampled[upper_edge_mask] = x_1_sampled
    e_1_sampled[~upper_edge_mask] = x_1_sampled
    
    return e_t, e_1_sampled

def _build_molecules(g, model, xt_traj=False, ep_traj=False):
    """Build final molecule objects"""
    # Set final values
    for feat in model.canonical_feat_order:
        if feat == 'e':
            g.edata[f'{feat}_1'] = g.edata[f'{feat}_t']
        else:
            g.ndata[f'{feat}_1'] = g.ndata[f'{feat}_t']
    
    g.edata['ue_mask'] = get_upper_edge_mask(g)
    g = g.to('cpu')
    
    # Build molecules
    return [SampledMolecule(g_i, model.atom_type_map,
                           ctmc_mol=model.parameterization == 'ctmc',
                           build_xt_traj=xt_traj,
                           build_ep_traj=ep_traj,
                           exclude_charges=model.exclude_charges)
            for g_i in dgl.unbatch(g)]

@torch.no_grad()
def sample_with_hybrid_guidance(good_model, bad_model, cfg_model, n_atoms, 
                             auto_guide_w={'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0},
                             cfg_guide_w={'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0},
                             interpolation_weight=0.5,  # 0: pure CFG, 1: pure auto-guidance
                             **kwargs):
    """
    Sample molecules using hybrid guidance with separate weights for each method.
    
    Args:
        good_model: Well-trained FlowMol model
        bad_model: Early checkpoint FlowMol model
        cfg_model: ClassifierFreeGuidance model
        n_atoms: Tensor of shape (batch_size,) containing number of atoms per molecule
        auto_guide_w: Dict with guidance weights for auto-guidance
        cfg_guide_w: Dict with guidance weights for CFG
        interpolation_weight: Weight between CFG (0.0) and auto-guidance (1.0)
        **kwargs: Other sampling parameters
    """
    # Initialize graph and features
    g = _create_batched_graph(n_atoms, kwargs.get('device', 'cuda'))
    g = good_model.sample_prior(g, *get_batch_idxs(g), get_upper_edge_mask(g))
    _initialize_features(g, good_model)
    
    # Setup integration
    t = torch.linspace(0, 1, kwargs.get('n_timesteps', 100), device=kwargs.get('device', 'cuda'))
    alpha_t = good_model.interpolant_scheduler.alpha_t(t)
    alpha_t_prime = good_model.interpolant_scheduler.alpha_t_prime(t)
    
    for s_idx in range(1, len(t)):
        # Get predictions
        good_pred, bad_pred = _get_model_predictions(
            good_model, bad_model, g, t, s_idx, 
            kwargs.get('properties_for_sampling'),
            kwargs.get('normalization_file_path'),
            kwargs.get('property_name')
        )
        
        # Get CFG predictions with its own guidance weight
        cfg_pred = cfg_model.vector_field(g, t_combined, 
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            guide_w=cfg_guide_w,  # Pass CFG-specific weights
            apply_softmax=True,
            remove_com=True
        )
        
        # Interpolate predictions with separate weights
        for feat in good_model.canonical_feat_order:
            if feat == 'x':
                # Auto-guidance velocity field with its weight
                vf_auto = bad_pred['x'] + auto_guide_w['x'] * (good_pred['x'] - bad_pred['x'])
                # CFG velocity field already includes its weight
                vf_cfg = cfg_pred['x']
                # Interpolate
                vf = (1 - interpolation_weight) * vf_cfg + interpolation_weight * vf_auto
                g.ndata['x_t'] = g.ndata['x_t'] + (t[s_idx] - t[s_idx-1]) * vf
            else:
                # Auto-guidance log probabilities with its weight
                log_p_auto = torch.log(bad_pred[feat] + 1e-10) + auto_guide_w[feat] * (
                    torch.log(good_pred[feat] + 1e-10) - torch.log(bad_pred[feat] + 1e-10)
                )
                # CFG log probabilities already include its weight
                log_p_cfg = torch.log(cfg_pred[feat] + 1e-10)
                # Interpolate
                log_p = (1 - interpolation_weight) * log_p_cfg + interpolation_weight * log_p_auto
                p_s_1 = F.softmax(log_p/good_model.vector_field.cat_temp_func(t[s_idx-1]), dim=-1)
                
                _update_categorical_feature(g, feat, p_s_1, good_model, s_idx, t, 
                    alpha_t, alpha_t_prime, kwargs.get('stochasticity'),
                    kwargs.get('high_confidence_threshold'))
    
    return _build_molecules(g, good_model, kwargs.get('xt_traj', False), kwargs.get('ep_traj', False))