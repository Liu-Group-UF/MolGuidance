import numpy as np
import pandas as pd
import pickle
import json
import argparse
import os
from rdkit import Chem
from str2prop_module import SimplePropertySampler, sample_n_atoms

PROPERTY_NAMES = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'cv']

def numpy_file_for_in_and_out_distribution_task(n_samples_outdis:int=1000,
                                                n_samples_indis:int=10_000,
                                                seed:int=42, 
                                                output_dir_outdis:str="out_of_distribution_sampling",
                                                output_dir_indis:str="in_distribution_sampling",
                                                dataset_name:str="qm9"):
    # "train_mol_idxs.npy" can be got by running "process_qm9_cond.py"
    #  which is also need to be run before training the model
    if dataset_name == "qm9":
        train_ind = np.load("data/qm9_raw/train_mol_idxs.npy")
        suppl = Chem.SDMolSupplier('data/qm9_raw/all_fixed_gdb9.sdf', removeHs=False, sanitize=False)
        no_atoms = [mol.GetNumAtoms() for mol in suppl]

        qm9_csv_file = "data/qm9_raw/gdb9.sdf.csv"
        df = pd.read_csv(qm9_csv_file)
        df['no_atoms'] = no_atoms
        train_df = df.iloc[train_ind]

        stats = lambda x: np.quantile(x, q=[0.5, 0.97, 0.99, 1])
        train_half_no_atoms = train_df.loc[:, 'no_atoms'].values
        # chosen_target_values tell what values are used for out-of-distribution task
        chosen_target_values = {}
        os.makedirs(output_dir_outdis, exist_ok=True)
        os.makedirs(output_dir_indis, exist_ok=True)

        for prop in PROPERTY_NAMES:
            prop_values = train_df.loc[:, prop].values
            # for out-of-distribution task
            median, low_prop, target, high_prop = stats(prop_values)
            chosen_target_values[prop] = target
            n_atoms = sample_n_atoms(prop_values, train_half_no_atoms, [low_prop, high_prop],
                                    n_samples=n_samples_outdis, random_seed=seed)
            np.save(f'{output_dir_outdis}/ood_n_atoms_{prop}.npy', n_atoms)
            
            # for in-distribution task
            sampler = SimplePropertySampler(train_half_no_atoms, prop_values)
            prop_samples, node_samples = sampler.sample(n_samples=n_samples_indis)
            np.save(f"{output_dir_indis}/train_half_sampled_values_{prop}.npy", prop_samples)
            np.save(f"{output_dir_indis}/train_half_sampled_no_atoms_{prop}.npy", node_samples)

    elif dataset_name == "qme14s":
        train_ind = np.load("data/qme14s_raw/train_mol_idxs.npy")
        suppl = Chem.SDMolSupplier('data/qme14s_raw/qme14s.sdf', removeHs=False, sanitize=False)
        no_atoms = [mol.GetNumAtoms() for mol in suppl]

        qme14s_csv_file = "data/qme14s_raw/dipole_moments_cleaned.csv"
        df = pd.read_csv(qme14s_csv_file)
        df['no_atoms'] = no_atoms
        train_df = df.iloc[train_ind]
        stats = lambda x: np.quantile(x, q=[0.5, 0.97, 0.99, 1])
        train_half_no_atoms = train_df.loc[:, 'no_atoms'].values
        chosen_target_values = {}
        os.makedirs(output_dir_outdis, exist_ok=True)
        os.makedirs(output_dir_indis, exist_ok=True)
        property_name = "mu" # only one property for qme14s
        prop_values = train_df.loc[:, property_name].values

        # for out-of-distribution task
        median, low_prop, target, high_prop = stats(prop_values)
        chosen_target_values[property_name] = target
        n_atoms = sample_n_atoms(prop_values, train_half_no_atoms, [low_prop, high_prop],
                                n_samples=n_samples_outdis, random_seed=seed)
        np.save(f'{output_dir_outdis}/ood_n_atoms_{property_name}.npy', n_atoms)

        # for in-distribution task
        sampler = SimplePropertySampler(train_half_no_atoms, prop_values)
        prop_samples, node_samples = sampler.sample(n_samples=n_samples_indis)
        np.save(f"{output_dir_indis}/train_half_sampled_values_{property_name}.npy", prop_samples)
        np.save(f"{output_dir_indis}/train_half_sampled_no_atoms_{property_name}.npy", node_samples)

    with open(f'{output_dir_outdis}/chosen_target_values.json', 'w') as f:
        json.dump(chosen_target_values, f, indent=4)

def parse_args():
    args = argparse.ArgumentParser(description="Generate numpy files for in and out distribution sampling tasks.")
    args.add_argument('--n_samples_outdis', type=int, default=1000,
                      help='Number of molecule being sampled for out-of-distribution task.')
    args.add_argument('--n_samples_indis', type=int, default=10_000,
                        help='Number of molecule being sampled for in-distribution task.')
    args.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling.')
    args.add_argument('--output_dir_outdis', type=str, default="out_of_distribution_sampling",
                        help='Directory to save out-of-distribution samples.')
    args.add_argument('--output_dir_indis', type=str, default="in_distribution_sampling",
                        help='Directory to save in-distribution samples.')
    args.add_argument('--dataset_name', type=str, default="qm9",
                        choices=["qm9", "qme14s"],
                        help='Dataset name to use for sampling.')

    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    numpy_file_for_in_and_out_distribution_task(
        n_samples_outdis=args.n_samples_outdis,
        n_samples_indis=args.n_samples_indis,
        seed=args.seed,
        output_dir_outdis=args.output_dir_outdis,
        output_dir_indis=args.output_dir_indis,
        dataset_name=args.dataset_name
    )
