import argparse
import atexit
import json
import pickle
import signal
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import tqdm
import yaml
from rdkit import Chem
from multiprocessing import Pool
import pandas as pd

from molguidance.data_processing.geom import MoleculeFeaturizer
from molguidance.utils.dataset_stats import compute_p_c_given_a

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(description='Process geometry')
    p.add_argument('--config', type=Path, help='config file path')
    p.add_argument('--chunk_size', type=int, default=1000, help='number of molecules to process at once')

    p.add_argument('--n_cpus', type=int, default=1, help='number of cpus to use when computing partial charges for confomers')
    # p.add_argument('--dataset_size', type=int, default=None, help='number of molecules in dataset, only used to truncate dataset for debugging')

    args = p.parse_args()

    return args

def process_all_molecules(dataset_config, args):
    raw_dir = Path(dataset_config['raw_data_dir']) 
    sdf_file = raw_dir / 'all_fixed_gdb9.sdf'
    # bad_mols_file = raw_dir / 'uncharacterized.txt'

    # # Get ids to skip
    # ids_to_skip = []
    # with open(bad_mols_file, 'r') as f:
    #     lines = f.read().split('\n')[9:-2]
    #     for x in lines:
    #         ids_to_skip.append(int(x.split()[0]) - 1)

    # Initialize storage
    mol_features = {
        'positions': [], 
        'atom_types': [], 
        'atom_charges': [],
        'bond_types': [], 
        'bond_idxs': [], 
        'smiles': []
    }
    valid_mol_idxs = []
    all_bond_order_counts = torch.zeros(5, dtype=torch.int64)
    
    # Process in chunks like original code
    mol_reader = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=True)
    mol_featurizer = MoleculeFeaturizer(dataset_config['atom_map'], n_cpus=args.n_cpus)
    
    all_mols = []
    chunk_indices = []  # Store original indices for each molecule in chunk
    chunk_size = args.chunk_size
    
    for global_idx, mol in enumerate(mol_reader):
        # if mol is None or global_idx in ids_to_skip:
        if mol is None:
            continue
        all_mols.append(mol)
        chunk_indices.append(global_idx)
        
        if len(all_mols) == chunk_size:
            positions, atom_types, atom_charges, bond_types, bond_idxs, num_failed, bond_counts, failed_chunk_idxs = mol_featurizer.featurize_molecules(all_mols)
            
            # Only store molecules that passed featurization
            for i in range(len(chunk_indices)):
                if i not in failed_chunk_idxs:  # If this index didn't fail
                    valid_mol_idxs.append(chunk_indices[i])
                    success_idx = i - sum(j < i for j in failed_chunk_idxs)  # Adjust index for successful cases
                    mol_features['positions'].append(positions[success_idx])
                    mol_features['atom_types'].append(atom_types[success_idx])
                    mol_features['atom_charges'].append(atom_charges[success_idx])
                    mol_features['bond_types'].append(bond_types[success_idx])
                    mol_features['bond_idxs'].append(bond_idxs[success_idx])
                    mol_features['smiles'].append(Chem.MolToSmiles(all_mols[i], isomericSmiles=True))
            
            all_bond_order_counts += bond_counts
            all_mols = []
            chunk_indices = []

    # Process remaining molecules
    if all_mols:
        remainder_size = len(all_mols)
        positions, atom_types, atom_charges, bond_types, bond_idxs, num_failed, bond_counts, failed_chunk_idxs = mol_featurizer.featurize_molecules(all_mols)
        
        for i in range(remainder_size):
            if i not in failed_chunk_idxs:
                valid_mol_idxs.append(chunk_indices[i])
                success_idx = i - sum(j < i for j in failed_chunk_idxs)
                mol_features['positions'].append(positions[success_idx])
                mol_features['atom_types'].append(atom_types[success_idx])
                mol_features['atom_charges'].append(atom_charges[success_idx])
                mol_features['bond_types'].append(bond_types[success_idx])
                mol_features['bond_idxs'].append(bond_idxs[success_idx])
                mol_features['smiles'].append(Chem.MolToSmiles(all_mols[i], isomericSmiles=True))
        
        all_bond_order_counts += bond_counts

    return mol_features, valid_mol_idxs, all_bond_order_counts

def process_split(split_features, split_df, split_name, split_bond_counts, dataset_config):

    # get processed data directory and create it if it doesn't exist
    output_dir = Path(dataset_config['processed_data_dir'])
    output_dir.mkdir(exist_ok=True) 

    all_positions = split_features['positions']
    all_atom_types = split_features['atom_types']
    all_atom_charges = split_features['atom_charges']
    all_bond_types = split_features['bond_types']
    all_bond_idxs = split_features['bond_idxs']
    all_smiles = split_features['smiles']
    all_bond_order_counts = split_bond_counts

    # get number of atoms in every data point
    n_atoms_list = [ x.shape[0] for x in all_positions ]
    n_bonds_list = [ x.shape[0] for x in all_bond_idxs ]

    # convert n_atoms_list and n_bonds_list to tensors
    n_atoms_list = torch.tensor(n_atoms_list)
    n_bonds_list = torch.tensor(n_bonds_list)

    # concatenate all_positions and all_features into single arrays
    all_positions = torch.concatenate(all_positions, dim=0)
    all_atom_types = torch.concatenate(all_atom_types, dim=0)
    all_atom_charges = torch.concatenate(all_atom_charges, dim=0)
    all_bond_types = torch.concatenate(all_bond_types, dim=0)
    all_bond_idxs = torch.concatenate(all_bond_idxs, dim=0)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's node features
    node_idx_array = torch.zeros((len(n_atoms_list), 2), dtype=torch.int32)
    node_idx_array[:, 1] = torch.cumsum(n_atoms_list, dim=0)
    node_idx_array[1:, 0] = node_idx_array[:-1, 1]

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's edge features
    edge_idx_array = torch.zeros((len(n_bonds_list), 2), dtype=torch.int32)
    edge_idx_array[:, 1] = torch.cumsum(n_bonds_list, dim=0)
    edge_idx_array[1:, 0] = edge_idx_array[:-1, 1]

    all_positions = all_positions.type(torch.float32)
    all_atom_charges = all_atom_charges.type(torch.int32)
    all_bond_idxs = all_bond_idxs.type(torch.int32)

    # propeties list:
    property_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                     'zpve', 'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom', 
                     'u298_atom', 'h298_atom', 'g298_atom']

    # extract propeties from the split_df and convert to tensor
    properties_tensor = torch.tensor(split_df[property_names].values, dtype=torch.float32)
    # normalize tensor
    properties_mean = properties_tensor.mean(dim=0)
    properties_std = properties_tensor.std(dim=0)
    # properties_normalized = (properties_tensor - properties_mean) / properties_std

    # Save the normalization parameters for later use
    normalization_params = {
        'mean': properties_mean,
        'std': properties_std,
        'property_names': property_names
    }

    # create a dictionary to store all the data
    data_dict = {
        'smiles': all_smiles,
        'positions': all_positions,
        'atom_types': all_atom_types,
        'atom_charges': all_atom_charges,
        'bond_types': all_bond_types,
        'bond_idxs': all_bond_idxs,
        'node_idx_array': node_idx_array,
        'edge_idx_array': edge_idx_array,
        'properties': properties_tensor,  
        'property_names': property_names,    # Add property names for reference
    }

    # determine output file name and save the data_dict there
    output_file = output_dir / f'{split_name}_processed.pt'
    torch.save(data_dict, output_file)

    # Save normalization parameters separately
    norm_params_file = output_dir / f'{split_name}_property_normalization.pt'
    torch.save(normalization_params, norm_params_file)

    # create histogram of number of atoms
    n_atoms, counts = torch.unique(n_atoms_list, return_counts=True)
    histogram_file = output_dir / f'{split_name}_n_atoms_histogram.pt'
    torch.save((n_atoms, counts), histogram_file)

    # compute the marginal distribution of atom types, p(a)
    p_a = all_atom_types.sum(dim=0)
    p_a = p_a / p_a.sum()

    # compute the marginal distribution of bond types, p(e)
    p_e = all_bond_order_counts / all_bond_order_counts.sum()

    # compute the marginal distirbution of charges, p(c)
    charge_vals, charge_counts = torch.unique(all_atom_charges, return_counts=True)
    p_c = torch.zeros(6, dtype=torch.float32)
    for c_val, c_count in zip(charge_vals, charge_counts):
        p_c[c_val+2] = c_count
    p_c = p_c / p_c.sum()

    # compute the conditional distribution of charges given atom type, p(c|a)
    p_c_given_a = compute_p_c_given_a(all_atom_charges, all_atom_types, dataset_config['atom_map'])

    # save p(a), p(e) and p(c|a) to a file
    marginal_dists_file = output_dir / f'{split_name}_marginal_dists.pt'
    torch.save((p_a, p_c, p_e, p_c_given_a), marginal_dists_file)

    # write all_smiles to its own file
    smiles_file = output_dir / f'{split_name}_smiles.pkl'
    with open(smiles_file, 'wb') as f:
        pickle.dump(all_smiles, f)

# Function to get features for a split
def get_split_features(indices, mol_features):
    split_features = {
        'positions': [mol_features['positions'][i] for i in indices],
        'atom_types': [mol_features['atom_types'][i] for i in indices],
        'atom_charges': [mol_features['atom_charges'][i] for i in indices],
        'bond_types': [mol_features['bond_types'][i] for i in indices],
        'bond_idxs': [mol_features['bond_idxs'][i] for i in indices],
        'smiles': [mol_features['smiles'][i] for i in indices]
    }
    return split_features

def get_bond_counts_with_unbonded(bond_types, atom_types):
    bond_counts = torch.zeros(5, dtype=torch.int64)
    
    # Count existing bonds
    unique_types, counts = torch.unique(bond_types, return_counts=True)
    for type_idx, count in zip(unique_types, counts):
        bond_counts[type_idx] += count
        
    # Calculate unbonded pairs for this molecule
    n_atoms = atom_types.shape[0]
    n_pairs = n_atoms * (n_atoms - 1) // 2  # Total possible pairs
    n_bonded_pairs = bond_types.shape[0]  # Use shape instead of len
    n_unbonded = n_pairs - n_bonded_pairs
    
    # Add unbonded count to first position
    bond_counts[0] = n_unbonded
    return bond_counts

# For processing a split:
def get_split_bond_counts(bond_types_list, atom_types_list):
    total_bond_counts = torch.zeros(5, dtype=torch.int64)
    
    for bond_types, atom_types in zip(bond_types_list, atom_types_list):
        mol_bond_counts = get_bond_counts_with_unbonded(bond_types, atom_types)
        total_bond_counts += mol_bond_counts
    
    return total_bond_counts


if __name__ == "__main__":

    # parse command-line args
    args = parse_args()

    # load config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_config = config['dataset']
    if dataset_config['dataset_name'] != 'qm9':
        raise ValueError('This script only works with the qm9 dataset')

    ##########3
    # this must be changed for the qm9 dataset
    ############3

    # get qm9 csv file as a pandas dataframe
    qm9_csv_file = Path(dataset_config['raw_data_dir']) / 'gdb9.sdf.csv'
    df = pd.read_csv(qm9_csv_file)

    mol_features, valid_mol_idxs, all_bond_order_counts = process_all_molecules(dataset_config, args)

    n_samples = len(valid_mol_idxs)
    n_train = 100000
    n_train_half = n_train // 2  # Split train data in half
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    # print the number of samples in each split
    print(f"Number of samples in train split: {n_train}")
    print(f"Number of samples in train_a split: {n_train_half}")
    print(f"Number of samples in train_b split: {n_train_half}")
    print(f"Number of samples in val split: {n_val}")
    print(f"Number of samples in test split: {n_test}")

    # First shuffle the valid_mol_idxs
    np.random.seed(42)  # For reproducibility
    shuffled_indices = np.random.permutation(n_samples)

    # Split the shuffled indices into train/val/test
    train_idx = shuffled_indices[:n_train]
    train_a_idx = shuffled_indices[:n_train_half]
    train_b_idx = shuffled_indices[n_train_half:n_train]
    val_idx = shuffled_indices[n_train:n_train+n_val]
    test_idx = shuffled_indices[n_train+n_val:]    

    # Get original molecule indices for each split
    train_mol_idx = [valid_mol_idxs[i] for i in train_idx]
    np.save(Path(dataset_config['raw_data_dir']) / 'train_mol_idxs.npy', train_mol_idx)
    train_a_mol_idx = [valid_mol_idxs[i] for i in train_a_idx]
    train_b_mol_idx = [valid_mol_idxs[i] for i in train_b_idx]
    val_mol_idx = [valid_mol_idxs[i] for i in val_idx]
    test_mol_idx = [valid_mol_idxs[i] for i in test_idx]

    # Get features for each split
    train_features = get_split_features(train_idx, mol_features)
    train_a_features = get_split_features(train_a_idx, mol_features)
    train_b_features = get_split_features(train_b_idx, mol_features)
    val_features = get_split_features(val_idx, mol_features)
    test_features = get_split_features(test_idx, mol_features)

    # Get properties for each split from the original dataframe
    train_df = df.iloc[train_mol_idx]
    train_a_df = df.iloc[train_a_mol_idx]
    train_b_df = df.iloc[train_b_mol_idx]
    val_df = df.iloc[val_mol_idx]
    test_df = df.iloc[test_mol_idx]

    # Get bond counts for each split
    train_bond_counts = get_split_bond_counts(
        train_features['bond_types'], 
        train_features['atom_types']
    )
    train_a_bond_counts = get_split_bond_counts(
        train_a_features['bond_types'], 
        train_a_features['atom_types']
    )
    train_b_bond_counts = get_split_bond_counts(
        train_b_features['bond_types'], 
        train_b_features['atom_types']
    )
    val_bond_counts = get_split_bond_counts(
        val_features['bond_types'], 
        val_features['atom_types']
    )
    test_bond_counts = get_split_bond_counts(
        test_features['bond_types'], 
        test_features['atom_types']
    )


    split_names = ['train_data', 'train_a_data', 'train_b_data', 'val_data', 'test_data']
    for split_features, split_df, split_name, split_bond_counts in zip(
        [train_features, train_a_features, train_b_features, val_features, test_features],
        [train_df, train_a_df, train_b_df, val_df, test_df],
        split_names,
        [train_bond_counts, train_a_bond_counts, train_b_bond_counts, val_bond_counts, test_bond_counts]
    ):
        process_split(split_features, split_df, split_name, split_bond_counts, dataset_config)

    




