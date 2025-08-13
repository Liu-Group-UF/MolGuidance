"""
Script for molecule sampling with Auto-Guidance

This script performs sampling of molecules using two FlowMol models:
- A well-trained model (good model) that produces high-quality samples
- An early checkpoint model (bad model) that acts as an unconditional baseline

By combining predictions from both models with guidance weights, we can control 
the generation process without needing classifier-free guidance training.
"""
import os
import argparse
import math
import numpy as np
import torch
from pathlib import Path
from rdkit import Chem

from molguidance.models.flowmol import FlowMol
from molguidance.models.classifier_free_guidance import CFGVectorField
from molguidance.models.ctmc_vector_field import PROPERTY_MAP
from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.analysis.metrics import SampleAnalyzer
from molguidance.data_processing.utils import build_edge_idxs, get_batch_idxs, get_upper_edge_mask, get_edge_batch_idxs
import torch.nn.functional as F
import dgl


def parse_arguments():
    parser = argparse.ArgumentParser(description="Molecule sampling with Auto-Guidance")
    
    # Model checkpoints
    parser.add_argument("--good_model_checkpoint", type=str, required=True,
                        help="Path to the well-trained model checkpoint")
    parser.add_argument("--bad_model_checkpoint", type=str, required=True,
                        help="Path to the early checkpoint model")
    
    # Sampling parameters
    parser.add_argument("--n_mols", type=int, default=1000,
                        help="Number of molecules to sample")
    parser.add_argument("--max_batch_size", type=int, default=128,
                        help="Maximum batch size for sampling")
    parser.add_argument("--n_atoms_per_mol", type=int, default=None,
                        help="Number of atoms per molecule (if fixed)")
    parser.add_argument("--n_timesteps", type=int, default=100,
                        help="Number of timesteps for sampling")
    
    # Trajectory and metrics options
    parser.add_argument("--xt_traj", action="store_true",
                        help="Store x_t trajectory")
    parser.add_argument("--ep_traj", action="store_true",
                        help="Store episode trajectory")
    parser.add_argument("--metrics", action="store_true",
                        help="Compute metrics")
    
    # Stochasticity and thresholds
    parser.add_argument("--stochasticity", type=float, default=None,
                        help="Stochasticity parameter")
    parser.add_argument("--hc_thresh", type=float, default=None,
                        help="High confidence threshold")
    
    # Property conditioning
    parser.add_argument("--property_name", type=str, default=None,
                        help="Property name")
    parser.add_argument("--properties_for_sampling", type=float, default=None,
                        help="Property value for conditioning")
    parser.add_argument("--normalization_file_path", type=str, default=None,
                        help="Path to property normalization file")
    parser.add_argument("--properties_handle_method", type=str, default="concatenate_sum",
                        choices=["concatenate", "sum", "multiply", "concatenate_sum", "concatenate_multiply"],
                        help="Method to handle properties")
    parser.add_argument("--multilple_values_file", type=str, default=None,
                        help="Path to numpy file containing multiple property values")
    parser.add_argument("--number_of_atoms", type=str, default=None,
                        help="Path to numpy file containing number of atoms in the molecule") 
    
    # Guidance weights
    parser.add_argument("--guide_w_x", type=float, default=2.0,
                        help="Guidance weight for coordinates")
    parser.add_argument("--guide_w_a", type=float, default=2.0,
                        help="Guidance weight for atom types")
    parser.add_argument("--guide_w_c", type=float, default=2.0,
                        help="Guidance weight for charges")
    parser.add_argument("--guide_w_e", type=float, default=2.0,
                        help="Guidance weight for bonds")

    # guidance other options
    parser.add_argument("--guidance_format", type=str, default="linear",
                        help="Guidance format: 'linear' or 'logistic'")
    parser.add_argument("--where_to_apply_guide", type=str, default="probabilities",
                        help="Where to apply guidance: 'probabilities' or 'rate_matrix'")

    # dfm options
    parser.add_argument("--dfm_type", type=str, default="campbell", choices=["campbell", "campbell_rate_matrix"],
                        help="DFM type")

    # Output
    parser.add_argument("--output_file", type=str, default="auto_guided_molecules.sdf",
                        help="Output SDF file path")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze generated molecules")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (or -1 for CPU)")
    
    # dataset name
    parser.add_argument("--dataset_name", type=str, default="qme14s",
                        help="Name of the dataset for loading normalization parameters")
    
    return parser.parse_args()

@torch.no_grad()
def sample_with_auto_guidance(good_model, bad_model, n_atoms, guide_w, 
                             n_timesteps=100, device="cuda:0",
                             stochasticity=8.0, high_confidence_threshold=0.9, 
                             xt_traj=False, ep_traj=False,
                             normalization_file_path=None, conditional_generation=True,
                             property_name=None, properties_for_sampling=None,
                             properties_handle_method='concatenate_sum',
                             multilple_values_to_one_property=None,
                             dfm_type='campbell', 
                             guidance_format='linear',  
                             where_to_apply_guide='probabilities',
                             dataset_name="qme14s"                             
                             ):
    """
    Sample molecules using auto-guidance from two models.
    
    Args:
        good_model: Well-trained FlowMol model
        bad_model: Early checkpoint FlowMol model (used as unconditional baseline)
        n_atoms: Tensor of shape (batch_size,) containing number of atoms per molecule
        guide_w: Weight for guidance, higher values follow good_model more closely
        n_timesteps: Number of integration steps
        device: Device to run the models on
        other args: Same as in FlowMol.sample()
    
    Returns:
        List of SampledMolecule objects
    """
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
    g = good_model.sample_prior(g, node_batch_idx, upper_edge_mask)

    # Get timepoints for integration
    t = torch.linspace(0, 1, n_timesteps, device=device)
    
    # Get alpha values
    alpha_t = good_model.interpolant_scheduler.alpha_t(t)
    alpha_t_prime = good_model.interpolant_scheduler.alpha_t_prime(t)
    
    # Initialize features
    for feat in good_model.canonical_feat_order:
        if feat == 'e':
            data_src = g.edata
        else:
            data_src = g.ndata
        data_src[f'{feat}_t'] = data_src[f'{feat}_0']
    
    # Setup visualization if needed
    if visualize:
        traj_frames = {}
        for feat in good_model.canonical_feat_order:
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
    
    # Set required attributes for both vector fields before starting integration
    # This is the key fix for the error - setting the same attributes on both models
    for model in [good_model, bad_model]:
        model.vector_field.properties_for_sampling = properties_for_sampling
        model.vector_field.property_name = property_name
        model.vector_field.conditional_generation = conditional_generation
        model.vector_field.normalization_file_path = normalization_file_path
        model.vector_field.training_mode = False
        model.vector_field.properties_handle_method = properties_handle_method
        model.vector_field.multilple_values_to_one_property = multilple_values_to_one_property
    
    # Integration loop
    for s_idx in range(1, t.shape[0]):
        s_i = t[s_idx]
        t_i = t[s_idx - 1]
        dt = s_i - t_i
        last_step = (s_idx == t.shape[0] - 1)
        
        assert properties_for_sampling is not None or multilple_values_to_one_property is not None

        # Prepare property information
        if multilple_values_to_one_property is not None:
            # Convert to tensor
            prop_tensor = torch.tensor(multilple_values_to_one_property, device=device).view(g.batch_size, 1)
            
            # Normalize if needed
            if normalization_file_path is not None:
                norm_params = torch.load(normalization_file_path)
                if dataset_name == "qm9":
                    assert property_name is not None, "property_name must be provided for qm9 dataset"
                    property_idx = PROPERTY_MAP.get(property_name)
                    mean = norm_params['mean'][property_idx].item()
                    std = norm_params['std'][property_idx].item()
                elif dataset_name == "qme14s":
                    mean = norm_params['mean'].item()
                    std = norm_params['std'].item()
                prop_tensor = (prop_tensor - mean) / std
             
        elif properties_for_sampling is not None:
            # For single property value
            if normalization_file_path is not None:
                norm_params = torch.load(normalization_file_path)
                if dataset_name == "qm9":
                    assert property_name is not None, "property_name must be provided for qm9 dataset"
                    property_idx = PROPERTY_MAP.get(property_name)
                    mean = norm_params['mean'][property_idx].item()
                    std = norm_params['std'][property_idx].item()
                elif dataset_name == "qme14s":
                    mean = norm_params['mean'].item()
                    std = norm_params['std'].item()
                normalized_value = (properties_for_sampling - mean) / std
                prop_value = normalized_value
            else:
                prop_value = properties_for_sampling
            
            prop_tensor = torch.full((g.batch_size, 1), prop_value, device=device)
            
        # Get property embedding
        prop_emb_good = good_model.property_embedder(prop_tensor)
        prob_emb_bad = bad_model.property_embedder(prop_tensor)
        
        t_combined_good = torch.cat([torch.full((g.batch_size, 1), t_i, device=device), prop_emb_good], dim=-1)
        t_combined_bad = torch.cat([torch.full((g.batch_size, 1), t_i, device=device), prob_emb_bad], dim=-1)
        
        good_pred = good_model.vector_field(g, t=t_combined_good,
                            node_batch_idx=node_batch_idx,
                            upper_edge_mask=upper_edge_mask,
                            apply_softmax=False,
                            remove_com=True)
        
        bad_pred = bad_model.vector_field(g, t=t_combined_bad,
                            node_batch_idx=node_batch_idx,
                            upper_edge_mask=upper_edge_mask,
                            apply_softmax=False,
                            remove_com=True)
        
        # Handle positions
        x_1_bad = bad_pred['x']
        x_1_good = good_pred['x']
        x_t = g.ndata['x_t']
        guide_w_pos = guide_w['x']
        
        # Compute velocity fields
        vf_bad = bad_model.vector_field.vector_field(x_t, x_1_bad, alpha_t[s_idx - 1][0], alpha_t_prime[s_idx - 1][0])
        vf_good = good_model.vector_field.vector_field(x_t, x_1_good, alpha_t[s_idx - 1][0], alpha_t_prime[s_idx - 1][0])
        
        # Apply guidance formula (standard CFG formula)
        vf = vf_bad + guide_w_pos * (vf_good - vf_bad)
        
        # Update positions
        g.ndata['x_t'] = x_t + dt * vf
        g.ndata['x_1_pred'] = (x_1_bad + guide_w_pos * (x_1_good - x_1_bad)).detach().clone()
        
        # Get temperature function from the good model
        cat_temp_func = good_model.vector_field.cat_temp_func
        
        # Handle categorical features
        for feat_idx, feat in enumerate(good_model.canonical_feat_order):
            if feat == 'x':
                continue
            
            guide_weight = guide_w[feat]  # Here 'feat' will be 'a', 'c', or 'e'
            
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            
            xt = data_src[f'{feat}_t'].argmax(-1)
            
            if feat == 'e':
                xt = xt[upper_edge_mask]

            # get temperature value 
            temperature = cat_temp_func(t_i)
            # Use the good model's Campbell step for transitions
            actual_stochasticity = stochasticity if stochasticity is not None else good_model.vector_field.eta
            actual_hc_thresh = high_confidence_threshold if high_confidence_threshold is not None else good_model.vector_field.hc_thresh

            if dfm_type == 'campbell':
                bad_val = F.softmax(bad_pred[feat],dim=-1)  # Get probabilities for bad model
                good_val = F.softmax(good_pred[feat], dim=-1)  # Get probabilities for good model
                # Get log probabilities
                log_p_bad = torch.log(bad_val)  
                log_p_good = torch.log(good_val)

                # Apply standard CFG formula in log space
                p_s_1 = torch.exp(log_p_bad + guide_weight * (log_p_good - log_p_bad))
                p_s_1 = p_s_1 / p_s_1.sum(dim=-1, keepdim=True)

                p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1)
                
                xt, x_1_sampled = good_model.vector_field.campbell_step(
                    p_1_given_t=p_s_1,
                    xt=xt,
                    stochasticity=actual_stochasticity,
                    hc_thresh=actual_hc_thresh,
                    alpha_t=alpha_t[s_idx - 1][feat_idx],
                    alpha_t_prime=alpha_t_prime[s_idx - 1][feat_idx],
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                    n_classes=good_model.vector_field.n_cat_feats[feat]+1,
                    mask_index=good_model.vector_field.mask_idxs[feat],
                    last_step=last_step,
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx
                )

            elif dfm_type =='campbell_rate_matrix':
                bad_val = bad_pred[feat]
                good_val = good_pred[feat]

                if where_to_apply_guide == "probabilities":
                    p_1_given_t_bad = F.softmax(bad_val, dim=-1)
                    p_1_given_t_good = F.softmax(good_val, dim=-1)
                elif where_to_apply_guide == "rate_matrix":
                    p_1_given_t_bad = F.softmax(bad_val / temperature, dim=-1)
                    p_1_given_t_good = F.softmax(good_val / temperature, dim=-1)

                xt, x_1_sampled = CFGVectorField.campbell_step_with_rate_matrix_cfg(
                    p_1_given_t_uncond=p_1_given_t_bad,
                    p_1_given_t_cond=p_1_given_t_good,
                    xt=xt,
                    stochasticity=actual_stochasticity,
                    alpha_t=alpha_t[s_idx - 1][feat_idx],
                    alpha_t_prime=alpha_t_prime[s_idx - 1][feat_idx],
                    dt=dt,
                    guide_weight=guide_weight,
                    n_classes=good_model.vector_field.n_cat_feats[feat]+1,
                    mask_index=good_model.vector_field.mask_idxs[feat],
                    uncond_val=bad_val,
                    cond_val=good_val,
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
        
        # Record visualization frames if needed
        if visualize:
            for feat in good_model.canonical_feat_order:
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
    for feat in good_model.canonical_feat_order:
        if feat == "e":
            g_data_src = g.edata
        else:
            g_data_src = g.ndata
        g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']
    
    g.edata['ue_mask'] = upper_edge_mask
    g = g.to('cpu')
    
    # Process visualization frames if needed
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
    
    # Build molecule objects
    molecules = []
    for mol_idx, g_i in enumerate(dgl.unbatch(g)):
        args = [g_i, good_model.atom_type_map]
        if visualize:
            args.append(reshaped_traj_frames[mol_idx])

        is_ctmc = good_model.parameterization == 'ctmc'
        molecules.append(SampledMolecule(*args,
            ctmc_mol=is_ctmc,
            build_xt_traj=xt_traj,
            build_ep_traj=ep_traj,
            exclude_charges=good_model.exclude_charges))
    
    return molecules

def sample_random_sizes_with_auto_guidance(good_model, 
                                            bad_model, 
                                            n_atoms, 
                                            guide_w, 
                                            n_timesteps, 
                                            device, 
                                            stochasticity, 
                                            high_confidence_threshold, 
                                            xt_traj, 
                                            ep_traj, 
                                            normalization_file_path, 
                                            conditional_generation,
                                            property_name, 
                                            properties_for_sampling, 
                                            properties_handle_method,
                                            multilple_values_to_one_property, 
                                            number_of_atoms,
                                            dfm_type='campbell', 
                                            guidance_format='linear',
                                            where_to_apply_guide='probabilities',
                                            dataset_name="qme14s"
                                            ):
    """
    Sample n_molecules with auto-guidance, using random molecule sizes from distribution.
    """
    if multilple_values_to_one_property is not None and properties_for_sampling is not None:
        raise ValueError('You can not provide both multilple_values_to_one_property and properties_for_sampling, only one of them should be provided')

    # get the number of atoms that will be in each molecules
    if number_of_atoms:
        atoms_per_molecule = torch.tensor(number_of_atoms).to(device)
    else:
        # Use the good model for sampling atom counts from training distribution
        atoms_per_molecule = good_model.sample_n_atoms(number_of_atoms).to(device)
    
    return sample_with_auto_guidance(
        good_model=good_model,
        bad_model=bad_model,
        n_atoms=atoms_per_molecule,
        guide_w=guide_w,
        n_timesteps=n_timesteps,
        device=device,
        stochasticity=stochasticity,
        high_confidence_threshold=high_confidence_threshold,
        xt_traj=xt_traj,
        ep_traj=ep_traj,
        normalization_file_path=normalization_file_path,
        conditional_generation=conditional_generation,
        property_name=property_name,
        properties_for_sampling=properties_for_sampling,
        properties_handle_method=properties_handle_method,
        multilple_values_to_one_property=multilple_values_to_one_property,
        dfm_type=dfm_type,
        guidance_format=guidance_format,
        where_to_apply_guide=where_to_apply_guide,
        dataset_name=dataset_name
    )

def main():
    args = parse_arguments()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Load models
    print(f"Loading good model from {args.good_model_checkpoint}")
    good_model = FlowMol.load_from_checkpoint(args.good_model_checkpoint)
    good_model.to(device)
    good_model.eval()
    
    print(f"Loading bad model from {args.bad_model_checkpoint}")
    bad_model = FlowMol.load_from_checkpoint(args.bad_model_checkpoint)
    bad_model.to(device)
    bad_model.eval()
    
    # Set up guidance weights
    guide_w = {
        'x': args.guide_w_x,
        'a': args.guide_w_a,
        'c': args.guide_w_c,
        'e': args.guide_w_e
    }
    print(f"Using guidance weights: {guide_w}\n")
    
    # Load multiple property values if specified
    multilple_values_to_one_property, number_of_atoms = None, None
    if args.multilple_values_file is not None:
        print(f"Loading multiple property values from {args.multilple_values_file}\n")
        multilple_values_to_one_property = np.load(args.multilple_values_file).tolist()
    if args.number_of_atoms is not None:
        print(f"Loading number of atoms from {args.number_of_atoms}\n")
        number_of_atoms = np.load(args.number_of_atoms).tolist()
    
    # Initialize analyzer if needed
    if args.analyze:
        analyzer = SampleAnalyzer()
    
    # Calculate number of batches
    n_batches = math.ceil(args.n_mols / args.max_batch_size)
    molecules = []
    
    print(f"Sampling {args.n_mols} molecules in {n_batches} batches\n")
    
    # Sampling loop
    for batch_idx in range(n_batches):
        print(f"Batch {batch_idx+1}/{n_batches}")
        n_mols_needed = args.n_mols - len(molecules)
        batch_size = min(n_mols_needed, args.max_batch_size)
        
        # Extract batch property values if using multiple values
        batch_property, batch_no_of_atoms = None, None
        if multilple_values_to_one_property is not None:
            batch_property = multilple_values_to_one_property[len(molecules): len(molecules) + batch_size]
        if number_of_atoms is not None:
            batch_no_of_atoms = number_of_atoms[len(molecules): len(molecules) + batch_size]
        
        # Sample molecules with guidance
        if args.n_atoms_per_mol is None:
            batch_molecules = sample_random_sizes_with_auto_guidance(
                good_model=good_model,
                bad_model=bad_model,
                n_atoms=batch_size,
                device=device,
                n_timesteps=args.n_timesteps,
                xt_traj=args.xt_traj,
                ep_traj=args.ep_traj,
                stochasticity=args.stochasticity,
                high_confidence_threshold=args.hc_thresh,
                conditional_generation=True,
                guide_w=guide_w,
                properties_for_sampling=args.properties_for_sampling,
                property_name=args.property_name,
                normalization_file_path=args.normalization_file_path,
                properties_handle_method=args.properties_handle_method,
                multilple_values_to_one_property=batch_property,
                number_of_atoms=batch_no_of_atoms,
                dfm_type=args.dfm_type,
                guidance_format=args.guidance_format,
                where_to_apply_guide=args.where_to_apply_guide,
                dataset_name=args.dataset_name
            )
        else:
            n_atoms = torch.full((batch_size,), args.n_atoms_per_mol, dtype=torch.long, device=device)
            batch_molecules = sample_with_auto_guidance(
                good_model=good_model,
                bad_model=bad_model,
                n_atoms=n_atoms,
                guide_w=guide_w,
                device=device,
                n_timesteps=args.n_timesteps,
                xt_traj=args.xt_traj,
                ep_traj=args.ep_traj,
                stochasticity=args.stochasticity,
                high_confidence_threshold=args.hc_thresh,
                conditional_generation=True,
                properties_for_sampling=args.properties_for_sampling,
                property_name=args.property_name,
                normalization_file_path=args.normalization_file_path,
                properties_handle_method=args.properties_handle_method,
                multilple_values_to_one_property=batch_property,
                dfm_type=args.dfm_type,
                guidance_format=args.guidance_format,
                where_to_apply_guide=args.where_to_apply_guide,
                dataset_name=args.dataset_name
            )
        
        molecules.extend(batch_molecules)
    
    # Analyze molecules if requested
    if args.analyze:
        print("Analyzing molecules...")
        analysis_results = analyzer.analyze(molecules, energy_div=False, functional_validity=True)
        print("Analysis results:")
        for metric, value in analysis_results.items():
            print(f"  {metric}: {value}")
    
    # Write molecules to SDF file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    print(f"Writing {len(molecules)} molecules to {args.output_file}")
    sdf_writer = Chem.SDWriter(args.output_file)
    sdf_writer.SetKekulize(False)
    valid_count = 0
    for mol in molecules:
        rdkit_mol = mol.rdkit_mol
        if rdkit_mol is not None:
            sdf_writer.write(rdkit_mol)
            valid_count += 1
    sdf_writer.close()
    
    print(f"Successfully wrote {valid_count} valid molecules to {args.output_file}")

if __name__ == "__main__":
    main()