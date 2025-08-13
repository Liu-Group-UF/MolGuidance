"""
Script for sampling molecules with classifier guidance in FlowMol.

This script demonstrates how to use the classifier guidance extension to 
guide molecule generation toward specific target properties.
"""
import os
import argparse
import math
import numpy as np
import torch
from pathlib import Path
from rdkit import Chem

from molguidance.models.classifier_guidance import ClassifierGuidanceFlowMol
from molguidance.analysis.metrics import SampleAnalyzer
from molguidance.classifier.train_classifier import PropertyPredictorModule

def parse_arguments():
    parser = argparse.ArgumentParser(description="Conditional molecule sampling with Guided FlowMol")
    
    # Model checkpoints
    parser.add_argument( "--flowmol_checkpoint", type=str, required=True,
        help="Path to the FlowMol checkpoint file")
    parser.add_argument("--classifier_checkpoint", type=str, required=True,
        help="Path to the property predictor checkpoint")
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
    parser.add_argument("--multilple_values_file", type=str, default=None,
                        help="Path to numpy file containing multiple property values")
    parser.add_argument("--number_of_atoms", type=str, default=None,
                        help="Path to numpy file containing number of atoms in the molecule")    

    # Guidance weights
    parser.add_argument("--guide_w_x", type=float, default=1.0,
                        help="Guidance weight for coordinates")
    parser.add_argument("--guide_w_a", type=float, default=1.0,
                        help="Guidance weight for atom types")
    parser.add_argument("--guide_w_c", type=float, default=1.0,
                        help="Guidance weight for charges")
    parser.add_argument("--guide_w_e", type=float, default=1.0,
                        help="Guidance weight for bonds")

    # Output
    parser.add_argument("--output_file", type=str, default="cfg_sampled_molecules.sdf",
                        help="Output SDF file path")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze generated molecules")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (or -1 for CPU)")
    
    return parser.parse_args()

def load_models(args):
    """
    Load FlowMol and classifier models.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (flowmol, classifier)
    """
    # Set the device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load FlowMol model
    print(f"Loading FlowMol model from {args.flowmol_checkpoint}")
    flowmol_ckpt = torch.load(args.flowmol_checkpoint, map_location=device)
    
    if 'hyper_parameters' in flowmol_ckpt:
        # Lightning checkpoint
        model_config = flowmol_ckpt['hyper_parameters']
        flowmol = ClassifierGuidanceFlowMol(**model_config)
        flowmol.load_state_dict(flowmol_ckpt['state_dict'])
    else:
        # Direct state dict
        raise ValueError("Expected a Lightning checkpoint for FlowMol model")
    
    flowmol = flowmol.to(device)
    flowmol.eval()
    
    # Load classifier model
    classifier = PropertyPredictorModule.load_from_checkpoint(args.classifier_checkpoint, strict=False)  # Ignore unexpected keys
    classifier = classifier.to(device)
    classifier.eval()
    
    # Set classifier in FlowMol model
    flowmol.set_classifier(classifier)
    
    return flowmol, classifier

def main():
    args = parse_arguments()
    
    # Load models
    flowmol, classifier = load_models(args)
    device = next(flowmol.parameters()).device
    
    # Set up guidance weights
    feature_guidance_scales = {
        'x': args.guide_w_x,
        'a': args.guide_w_a,
        'c': args.guide_w_c,
        'e': args.guide_w_e
    }
    print(f"Using feature guidance scales: {feature_guidance_scales}")
    
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
        # print(f"Batch {batch_idx+1}/{n_batches}")
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
            batch_molecules = flowmol.sample_random_sizes(
                batch_size,
                target_function=None, # Use default MSE function
                device=device,
                n_timesteps=args.n_timesteps,
                xt_traj=args.xt_traj,
                ep_traj=args.ep_traj,
                stochasticity=args.stochasticity,
                high_confidence_threshold=args.hc_thresh,
                properties_for_sampling=args.properties_for_sampling,
                property_name=args.property_name,
                normalization_file_path=args.normalization_file_path,
                multilple_values_to_one_property=batch_property,
                number_of_atoms=batch_no_of_atoms,
                feature_guidance_scales=feature_guidance_scales,
                conditional_generation=False # very important to set this to False since we are using classifier guidance
            )
        else:
            n_atoms = torch.full((batch_size,), args.n_atoms_per_mol, dtype=torch.long, device=device)
            batch_molecules = flowmol.sample(
                n_atoms,
                target_function=None, 
                device=device,
                n_timesteps=args.n_timesteps,
                xt_traj=args.xt_traj,
                ep_traj=args.ep_traj,
                stochasticity=args.stochasticity,
                high_confidence_threshold=args.hc_thresh,
                properties_for_sampling=args.properties_for_sampling,
                property_name=args.property_name,
                normalization_file_path=args.normalization_file_path,
                multilple_values_to_one_property=batch_property,
                feature_guidance_scales=feature_guidance_scales,
                conditional_generation=False, 
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