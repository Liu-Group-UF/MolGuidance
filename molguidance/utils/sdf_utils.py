import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
# from .gaussian_utils import extract_last_optimized_geometry
from rdkit.Chem import rdDetermineBonds
from collections import defaultdict, Counter
from scipy.stats import entropy
import math

def write_xyz(geometry, xyz_filename):
    with open(xyz_filename, 'w') as f:
        f.write(f"{len(geometry)}\n")
        f.write("Extracted from Gaussian output\n")
        f.write("\n".join(geometry) + "\n")

def xyz_to_rdkit(xyz_file):
    raw_mol = Chem.MolFromXYZFile(xyz_file)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol)#,charge=0)
    return mol

def get_mol(gaussian_output):
    xyz_filename = "temp.xyz"
    geometry = extract_last_optimized_geometry(gaussian_output)
    write_xyz(geometry, xyz_filename)
    mol = xyz_to_rdkit(xyz_filename)
    return mol

def create_relaxed_sdf(output_log_files, output_sdf_file='relaxed.sdf'):
    sdf_writer = Chem.SDWriter(str(output_sdf_file))
    sdf_writer.SetKekulize(False)
    keep_inds = []
    mols = []
    for i, gaussian_log in enumerate(output_log_files):
        try:
            mol = get_mol(gaussian_log)
        except:
            continue
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        keep_inds.append(i)
        sdf_writer.write(mol)
        mols.append(mol)
    return mols, keep_inds

def get_rdkit_valid(sdf_file, n_mols=1000):
    from rdkit import RDLogger, Chem
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
    suppl = [_ for i, _ in enumerate(suppl) if i < n_mols]
    sdf_valid_index = [i for i, mol in enumerate(suppl) if mol is not None]
    return len(sdf_valid_index)/n_mols

def remove_mol_with_multifrags(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
    to_skip = []
    valid_mols = []
    for i, mol in enumerate(suppl):
        if mol is not None:
            fragments = Chem.GetMolFrags(mol, asMols=True)
            num_fragments = len(fragments)
            if num_fragments > 1:
                to_skip.append(i)
            else:
                valid_mols.append(mol)
        else:
            to_skip.append(i)
    return valid_mols, to_skip

def get_complete_single_molecule(sdf_file, n_mols=10000):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
    suppl = [_ for i, _ in enumerate(suppl) if i < n_mols]
    valid_inds = []
    for i, mol in enumerate(suppl):
        if mol is not None:
            fragments = Chem.GetMolFrags(mol, asMols=True)
            num_fragments = len(fragments)
            if num_fragments == 1:
                valid_inds.append(i)
    return valid_inds

def molecules_to_sdf(molecules, output_sdf_file):
    sdf_writer = Chem.SDWriter(str(output_sdf_file))
    sdf_writer.SetKekulize(False)
    count = 0
    indices = []  # indices for sanitized molecules  
    for i, mol in enumerate(molecules):
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        count += 1
        indices.append(i)
        sdf_writer.write(mol)
    sdf_writer.close()
    # return count, indices

from posebusters import PoseBusters
from pathlib import Path
import numpy as np
def get_pb_valid_results(sdf_file):
    RDLogger.DisableLog('rdApp.*')
    raw_mols = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    rdkit_valid_mols, valid_indices = [], []
    for i, mol in enumerate(raw_mols):
        if mol is not None:
            rdkit_valid_mols.append(mol)
            valid_indices.append(i)
    
    csv_file = sdf_file.replace('.sdf', '.csv')
    if not Path(csv_file).exists():
        molecules_to_sdf(rdkit_valid_mols, sdf_file)
        pred_file = Path(sdf_file)
        buster = PoseBusters(config="mol")
        df = buster.bust([pred_file], None, None, full_report=True)

        df['all_cond'] = True
        mask = True
        for col_ind in range(0, 10):
            mask &= df.iloc[:, col_ind]
        df['all_cond'] = df['all_cond'][mask]
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)

    pb_valid_inds = np.where(df['all_cond'].values == True)[0]
    pb_inds_remapped = np.array(valid_indices)[pb_valid_inds]
    mols_pb = np.array(rdkit_valid_mols)[pb_valid_inds]
    return pb_inds_remapped, mols_pb

def run_pb_valid_from_sdf(sdf_file):
    pred_file = Path(sdf_file)
    buster = PoseBusters(config="mol")
    csv_file = sdf_file.replace('.sdf', '.csv')
    df = buster.bust([pred_file], None, None, full_report=True)
    df.to_csv(csv_file, index=False)
    return df 
        
def get_pb_valid_ratio(sdf_file):
    csv_file = sdf_file.replace('.sdf', '.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Csv file {csv_file} not found.")
    df = pd.read_csv(csv_file, index_col=False)
    df['all_cond'] = True
    mask = True
    # print(df.columns)
    col_names = ['sanitization', 'all_atoms_connected', 'bond_lengths', 'bond_angles',
                'aromatic_ring_flatness', 'double_bond_flatness', 'internal_energy',
                'internal_steric_clash', 'passes_valence_checks', 'passes_kekulization']
    for col_name in col_names:
        mask &= df.loc[:, col_name]
    df['all_cond'] = df['all_cond'][mask]
    df.to_csv(csv_file, index=False)
    pb_valid_inds = np.where(df['all_cond'].values == True)[0]
    pb_valid_ratio = len(pb_valid_inds) / len(df) 
    return pb_valid_ratio

def check_valency_charge_balance(file_path, condition=1):
    suppl = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
    
    valid_mol_indices = []
    valid_mols = []
    valid_mol_count = 0
    valid_atom_count = 0
    total_mol_count = 0
    total_atom_count = 0

    
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        total_charge = 0
        total_mol_count += 1
        atoms = mol.GetAtoms()
        total_atom_count += len(atoms)

        atom_validities = []
        for atom in atoms:
            symbol = atom.GetSymbol()
            valence = atom.GetValence(Chem.ValenceType.EXPLICIT)
            charge = atom.GetFormalCharge()
            total_charge += charge
            expected = Chem.GetPeriodicTable().GetDefaultValence(symbol)
            deficit = valence - expected
            atom_valid = (deficit - charge == 0)
            atom_validities.append(atom_valid)
        
        valid_atom_count += sum(atom_validities)
        charge_cond =  total_charge == 0 if condition == 1 else 1
        if all(atom_validities) and charge_cond:
            valid_mol_count += 1
            valid_mol_indices.append(i)
            valid_mols.append(mol)
    
    atom_ratio = valid_atom_count / total_atom_count if total_atom_count > 0 else 0.0
    mol_ratio = valid_mol_count / total_mol_count if total_mol_count > 0 else 0.0

    return valid_mol_indices, mol_ratio, atom_ratio, valid_mols



def get_mol_stable(sdf_file, n_mols):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    suppl = [_ for i, _ in enumerate(suppl) if i < n_mols]
    valid_inds = []
    for i, mol in enumerate(suppl):
        if mol is not None:
            if check_mol_stable(mol):
                valid_inds.append(i)
    return valid_inds
            
def detect_undercoordinated_atoms(mol):
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()
        expected = Chem.GetPeriodicTable().GetDefaultValence(symbol)
        deficit = valence - expected
        if deficit - charge != 0:
            return True
    return False

VALENCE_ELECTRONS = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 1, "Ca": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
    "Rb": 1, "Sr": 2, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,
    "Cs": 1, "Ba": 2, "Tl": 3, "Pb": 4, "Bi": 5, "Po": 6, "At": 7, "Rn": 8,
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, 
    "Cu": 11, "Zn": 12
}
def has_even_valence_electrons(mol):
    total_valence = int(sum(VALENCE_ELECTRONS.get(atom.GetSymbol(), 0) for atom in mol.GetAtoms()))
    return total_valence % 2 == 0

def get_close_shell_ratio_from_sdf(file_path):
    suppl = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
    results = []
    even_count = 0
    # total_count = len(suppl)
    total_count = 0
    
    for i, mol in enumerate(suppl):
        if mol is None:
            results.append((i + 1, "Invalid Molecule"))
        else:
            is_even = has_even_valence_electrons(mol)
            results.append((i + 1, "Even" if is_even else "Odd"))
            if is_even:
                even_count += 1
            total_count += 1
    
    ratio = even_count / total_count if total_count > 0 else 0
    return ratio

def get_closed_shell_valid(file_path, n_mols):
    suppl = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
    suppl = [_ for i, _ in enumerate(suppl) if i < n_mols]
    valid_inds = []
    for i, mol in enumerate(suppl):
        if mol is not None:
            is_even = has_even_valence_electrons(mol)
            if is_even:
                valid_inds.append(i)
    return valid_inds

def get_uniqueness_rate(sdf_file, n_raw=10000, sanitize=True):
    """Extract SMILES from an SDF file."""
    RDLogger.DisableLog('rdApp.*')
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=sanitize)
    smiles_list = [Chem.MolToSmiles(mol) for mol in supplier if mol is not None]
    n_unique = len(set(smiles_list)) # unique and valid rdkit mols
    uniqueness_rate = n_unique / n_raw 
    return smiles_list, uniqueness_rate

def calculate_novelty_rate(generated_smiles, training_smiles):
    """Calculate uniqueness and novelty rates."""
    unique_smiles = set(generated_smiles)
    novel_smiles = unique_smiles - set(training_smiles)

    novelty_rate = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0

    return novelty_rate

def compute_all_standard_metrics(sdf_file, n_mols=10000):
    valid_mol_indices, mol_ratio, atom_ratio, valid_mols = check_valency_charge_balance(sdf_file)
    rdkit_valid = get_rdkit_valid(sdf_file, n_mols=n_mols)
    # _, valid_and_unique = get_uniqueness_rate(sdf_file, n_raw=n_mols)
    valid_and_unique = None
    close_shell_ratio = get_close_shell_ratio_from_sdf(sdf_file)
    return atom_ratio, mol_ratio, rdkit_valid, valid_and_unique, close_shell_ratio

def check_mol_stable(mol):
    atoms = mol.GetAtoms()
    atom_validities = []
    for atom in atoms:
        symbol = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()
        expected = Chem.GetPeriodicTable().GetDefaultValence(symbol)
        deficit = valence - expected
        atom_valid = (deficit - charge == 0)
        atom_validities.append(atom_valid)
    if all(atom_validities):
        return True
    return False

def extract_bond_distances(sdf_path):
    bond_dists = defaultdict(list)
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    
    for mol in suppl:
        if mol is None:
            continue
        conf = mol.GetConformer()
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            idx1, idx2 = a1.GetIdx(), a2.GetIdx()
            sym_pair = tuple(sorted((a1.GetSymbol(), a2.GetSymbol())))
            pos1 = conf.GetAtomPosition(idx1)
            pos2 = conf.GetAtomPosition(idx2)
            dist = pos1.Distance(pos2)
            bond_dists[sym_pair].append(dist)
    
    return bond_dists

def extract_bond_angles(sdf_path):
    angle_dists = defaultdict(list)
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    for mol in suppl:
        if mol is None:
            continue
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            neighbors = atom.GetNeighbors()
            if len(neighbors) < 2:
                continue
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a1 = neighbors[i]
                    a3 = neighbors[j]
                    idx1 = a1.GetIdx()
                    idx2 = atom.GetIdx()
                    idx3 = a3.GetIdx()
                    pos1 = conf.GetAtomPosition(idx1)
                    pos2 = conf.GetAtomPosition(idx2)
                    pos3 = conf.GetAtomPosition(idx3)

                    v1 = np.array([pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z])
                    v2 = np.array([pos3.x - pos2.x, pos3.y - pos2.y, pos3.z - pos2.z])

                    # Compute angle in degrees
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

                    sym_triplet = tuple(sorted([a1.GetSymbol(), atom.GetSymbol(), a3.GetSymbol()]))
                    angle_dists[sym_triplet].append(angle)

    return angle_dists

def sdf_to_smiles_list(sdf_path):
    """
    Reads molecules from an SDF file and returns a list of canonical SMILES strings.
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    smiles_list = []
    for mol in suppl:
        if mol is None:  # skip parsing failures
            continue
        # Generate canonical SMILES
        smi = Chem.MolToSmiles(mol, canonical=True)
        smiles_list.append(smi)
    return smiles_list

def calculate_bond_entropy(mols):
    # initialize global bond‐type counters
    total_bond_types = {'SINGLE': 0, 'DOUBLE': 0, 'TRIPLE': 0, 'AROMATIC': 0}
    # per‐molecule bond counts
    mol_bond_counts = []

    for mol in mols:
        curr_bonds = dict.fromkeys(total_bond_types, 0)
        for bond in mol.GetBonds():
            btype = str(bond.GetBondType())
            if btype in curr_bonds:
                curr_bonds[btype] += 1
                total_bond_types[btype] += 1
        mol_bond_counts.append(curr_bonds)

    # build DataFrame and sum counts
    df_bonds = pd.DataFrame(mol_bond_counts)
    bond_keys = list(total_bond_types.keys())
    bond_counts = df_bonds[bond_keys].sum(axis=0)

    # compute entropy
    total = bond_counts.sum()
    if total > 0:
        bond_probs = bond_counts / total
        bond_entropy = -np.sum(bond_probs * np.log2(bond_probs + 1e-10))
    else:
        bond_entropy = 0.0

    return bond_entropy, total_bond_types

def get_bond_entropy(sdf_files, titles):
    RDLogger.DisableLog('rdApp.*')
    # sdf_files : list of str (file paths)
    # titles: method names corresponding to sdf_files

    bond_entropies = []
    bond_type_distributions = []

    for sdf_file in sdf_files:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
        mols = [mol for mol in suppl if mol is not None]
        bond_entropy, bond_types = calculate_bond_entropy(mols)
        bond_entropies.append(bond_entropy)
        bond_type_distributions.append(bond_types)

    metrics_data = []
    
    for entropy, bond_types, title in zip(bond_entropies, bond_type_distributions, titles):
        bond_type_str = ', '.join([f'{k}: {v}' for k, v in bond_types.items()])
        metrics_data.append({
            'Method': title,
            'Bond Entropy': f'{entropy:.3f}',
            'Bond Type Distribution': bond_type_str
        })
    
    df = pd.DataFrame(metrics_data)
    return df

def load_molecules(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True) # must be sanitized before calling many diversity analyzing functions
    mols = [mol for mol in suppl if mol is not None]
    return mols

def get_element_entropy(sdf_files, titles):
    RDLogger.DisableLog('rdApp.*')
    # sdf_files : list of str (file paths)
    # titles: method names corresponding to sdf_files

    # Analyze each SDF file
    element_counts_all = []
    
    for sdf_file in sdf_files:
        element_counter = Counter()
        mol_count = 0
        
        # Read molecules from SDF file
        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
        
        for mol in supplier:
            if mol is not None:
                mol_count += 1
                # Count atoms in this molecule
                for atom in mol.GetAtoms():
                    element = atom.GetSymbol()
                    element_counter[element] += 1

        element_counts_all.append(dict(element_counter))

    metrics_data = []
    
    for counts, title in zip(element_counts_all, titles):
        total = sum(counts.values())
        proportions = np.array([c/total for c in counts.values()])
        
        # Shannon entropy (bits) - higher = more diverse
        shannon_entropy = entropy(proportions, base=2)
    
        metrics_data.append({
            'Method': title,
            'Shannon Entropy': f'{shannon_entropy:.3f}',
        })
    
    df = pd.DataFrame(metrics_data)
    return df

def calculate_scaffold_diversity(sdf_file):
    RDLogger.DisableLog('rdApp.*')
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
    mols = [mol for mol in suppl if mol is not None]

    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    # Generate scaffolds
    scaffolds = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    scaffold_smiles = [Chem.MolToSmiles(scaffold) for scaffold in scaffolds]
    
    # Count unique scaffolds
    unique_scaffolds = set(scaffold_smiles)
    scaffold_diversity = len(unique_scaffolds) / len(mols)
    
    return scaffold_diversity

def get_scaffold_diversity(sdf_files, titles):
    # sdf_files : list of str (file paths)
    # titles: method names corresponding to sdf_files

    scaffold_diversities = []
    
    for sdf_file in sdf_files:
        diversity = calculate_scaffold_diversity(sdf_file)
        scaffold_diversities.append(diversity)

    metrics_data = []
    
    for diversity, title in zip(scaffold_diversities, titles):
        metrics_data.append({
            'Method': title,
            'Scaffold Diversity': f'{diversity:.3f}',
        })
    
    df = pd.DataFrame(metrics_data)
    return df