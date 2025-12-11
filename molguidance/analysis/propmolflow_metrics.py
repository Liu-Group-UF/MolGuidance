import numpy as np
import pandas as pd
from posebusters import PoseBusters
from pathlib import Path
from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')

def get_rdkit_valid(sdf_file, n_mols=10000):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
    suppl = [_ for i, _ in enumerate(suppl) if i < n_mols]
    sdf_valid_index = [i for i, mol in enumerate(suppl) if mol is not None]
    return len(sdf_valid_index)/n_mols

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

def get_uniqueness_rate(sdf_file, n_raw=10000):
    """Extract SMILES from an SDF file."""
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
    smiles_list = [Chem.MolToSmiles(mol) for mol in supplier if mol is not None]
    n_unique = len(set(smiles_list)) # unique and valid rdkit mols
    uniqueness_rate = n_unique / n_raw 
    return smiles_list, uniqueness_rate

def calculate_novelty_rate(generated_smiles, training_smiles):
    """Calculate uniqueness and novelty rates."""
    unique_smiles = set(generated_smiles)
    novel_smiles = unique_smiles - set(training_smiles)

    novelty_rate = len(novel_smiles) / len(generated_smiles) if generated_smiles else 0

    return novelty_rate

expected_valences = {
    'H': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'P': 3,
    'S': 2,
    'Cl': 1,
    'Br': 1,
    'I': 1,
    'Al': 3,
    'Si': 4,
    'As':3,
    'Se':2,
    'B': 3
}
alternative_valences = {
    'S': 6,
    'Si': 2,
    'As': 5,
    'Se': 6
}
def check_valency_charge_balance(file_path):
    suppl = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
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
            val = atom.GetExplicitValence()
            charge = atom.GetFormalCharge()
            effective_valence = val - charge

            valid = False
            if symbol in expected_valences:
                expected = expected_valences[symbol]
                valid = (effective_valence == expected)

                if not valid and symbol in alternative_valences:
                    alt = alternative_valences[symbol]
                    valid = (effective_valence == alt)
                if not valid and symbol == 'Se':
                    valid = effective_valence == 4
                    
            else:
                valid = False

            atom_validities.append(valid)
        
        valid_atom_count += sum(atom_validities)
        charge_cond =  total_charge == 0
        if all(atom_validities) and charge_cond: 
            valid_mol_count += 1
    atom_ratio = valid_atom_count / total_atom_count if total_atom_count > 0 else 0.0
    mol_ratio = valid_mol_count / total_mol_count if total_mol_count > 0 else 0.0

    return mol_ratio, atom_ratio

def compute_all_standard_metrics(sdf_file, n_mols=10000):
    mol_ratio, atom_ratio = check_valency_charge_balance(sdf_file)
    rdkit_valid = get_rdkit_valid(sdf_file, n_mols=n_mols)
    smiles_list, valid_and_unique = get_uniqueness_rate(sdf_file, n_raw=n_mols)
    close_shell_ratio = get_close_shell_ratio_from_sdf(sdf_file)
    results = {}
    results['rdkit_valid'] = rdkit_valid
    results['valid_and_unique'] = valid_and_unique
    results['close_shell_ratio'] = close_shell_ratio
    results['atom_stable'] = atom_ratio
    results['mol_stable'] = mol_ratio
    return results

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
    return indices

def get_pb_valid_results(raw_mols, sdf_file):
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