import os
import torch 
import yaml
# import ase.io
import dgl
import math
import glob
from typing import List, Union
from tqdm import tqdm
from rdkit import Chem

import numpy as np
from molguidance.data_processing.geom import MoleculeFeaturizer
from molguidance.property_regressor.train_regressor import GVPRegressorModule
from molguidance.data_processing.dataset import collate
from pathlib import Path


# These values are close to [property_mean] + 3 * [property_std] from QM9 full dataset
# chosen_values_by_property = {
# "gap": 0.39, "alpha": 100, "mu":7.3, "homo": -0.17, "lumo": 0.15, "cv": 44, "cv_low": 20
# }
# Now we use the 99% percentile values from the QM9 training dataset
chosen_values_by_property = {'alpha': 93.31010000000002,
 'gap': 0.3436010000000002,
 'homo': -0.1757,
 'lumo': 0.091,
 'mu': 6.742203000000001,
 'cv': 41.20805000000001}

allowed_properties = ['mu', 'lumo', 'cv', 'homo', 'gap', 'alpha', 'dipole_total']
Hartree2eV=27.2114

def load_model_from_ckpt(ckpt, method=None):
    """Load PropFlowMol model from a checkpoint file."""
    from molguidance.models.flowmol import FlowMol
    if not method:
        model = FlowMol.load_from_checkpoint(ckpt)
    elif method == 'cfg':
        from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
        model = ClassifierFreeGuidance.load_from_checkpoint(ckpt)
    elif method == 'mg':
        from molguidance.models.model_guidance_scale import ModelGuidance
        model = ModelGuidance.load_from_checkpoint(ckpt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model

def sample_molecules(model,
                    n_mols, 
                    property_name,
                    properties_handle_method,
                    multiple_values_to_one_property=None,
                    number_of_atoms = None,
                    properties_for_sampling = None,
                    normalization_file_path=None,
                    n_timesteps=100,
                    stochasticity=None, 
                    max_batch_size=128, 
                    xt_traj=False, 
                    ep_traj=False,
                    high_confidence_threshold=None,
                    training_mode=False,
                    device='cuda'
                    ):
    assert property_name in allowed_properties, f"Property {property_name} is not supported."
    assert (properties_for_sampling is None) ^ (multiple_values_to_one_property is None)      

    n_batches = math.ceil(n_mols / max_batch_size)
    molecules = []
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
            properties_for_sampling=properties_for_sampling,
            training_mode = training_mode,
            property_name = property_name,
            normalization_file_path = normalization_file_path,
            properties_handle_method = properties_handle_method,
            multilple_values_to_one_property = batch_property,
            number_of_atoms = batch_no_atoms,
        )
        molecules.extend(batch_molecules)
    return molecules

from molguidance.analysis.metrics import SampleAnalyzer
def get_flowmol_metrics(sampledmolecules, energy_div=True, functional_validity=True):
    """Compute the metrics for the generated molecules, including:
    Atomic stability
    """
    analyzer = SampleAnalyzer()
    metrics_dict = analyzer.analyze(sampledmolecules, energy_div=energy_div, 
                                    functional_validity=functional_validity)
    return metrics_dict
    
def molecules_to_sdf(molecules, output_sdf_file, property_values=None):
    sdf_writer = Chem.SDWriter(str(output_sdf_file))
    sdf_writer.SetKekulize(False)

    for mol in molecules:
        rdkit_mol = mol.rdkit_mol
        sdf_writer.write(rdkit_mol)
    if property_values:
        base, _ = os.path.splitext(output_sdf_file)  # drops only the final extension
        npy_file = base + ".npy" 
        np.save(npy_file, property_values)
    return 
    
def get_model_ckpts(folder_path, ckpt_file="epoch*.ckpt"):
    file_list = glob.glob(f"{folder_path}/**/checkpoints/{ckpt_file}", recursive=True)
    files_sorted = sorted(file_list, key=lambda f: os.path.getctime(f), reverse=True)
    return files_sorted

PROPERTY_MAP = {
        'A': 0, 'B': 1, 'C': 2, 'mu': 3, 'alpha': 4, 
        'homo': 5, 'lumo': 6, 'gap': 7, 'r2': 8,
        'zpve': 9, 'u0': 10, 'u298': 11, 'h298': 12, 
        'g298': 13, 'cv': 14, 'u0_atom': 15, 'u298_atom': 16,
        'h298_atom': 17, 'g298_atom': 18
    }

class MoleculePredictor:
    def __init__(self, checkpoint_path: str, config_path: str, property_name: str):
        """
        Initialize the predictor with a trained model
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file used during training
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load model
        self.model = GVPRegressorModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize featurizer
        self.featurizer = MoleculeFeaturizer(self.config['dataset']['atom_map'])
        self.successful_indices = []  # Add this to track which molecules were successfully processed

        # Load normalization parameters if they exist
        norm_params_path = Path(self.config['dataset']['processed_data_dir']) / 'train_data_property_normalization.pt'
        self.dataset_name = self.config['dataset'].get('dataset_name', 'qm9')
        if self.dataset_name == 'qm9':
            self.property_idx = PROPERTY_MAP[property_name]
        if norm_params_path.exists():
            self.norm_params = torch.load(norm_params_path)
        else:
            self.norm_params = None

    def process_sdf(self, sdf_path: str, properties=None) -> tuple:
        """Process molecules from an SDF file"""
        mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        graphs = []
        self.successful_indices = []  # Reset successful indices
        
        for idx, mol in enumerate(tqdm(mol_supplier, desc="Processing molecules")):
            if mol is None:
                continue
            graph = self._mol_to_graph(mol)
            if graph is not None:
                graphs.append(graph)
                self.successful_indices.append(idx)
        
        # Filter properties if provided
        filtered_properties = None
        if properties is not None and isinstance(properties, (list, np.ndarray)):
            filtered_properties = [properties[idx] for idx in self.successful_indices]
        elif properties is not None and isinstance(properties, (float, int)):
            filtered_properties = [properties] * len(graphs)
                
        return graphs, filtered_properties

    def process_xyz_files(self, xyz_paths: List[str], properties=None) -> tuple:
        """Process molecules from XYZ files"""
        graphs = []
        self.successful_indices = []  # Reset successful indices

        for idx, xyz_path in enumerate(tqdm(xyz_paths, desc="Processing molecules")):
            # Read XYZ file using ASE
            mol = ase.io.read(xyz_path)
            # Convert to RDKit mol
            atoms = mol.get_chemical_symbols()
            positions = mol.get_positions()
            
            # Create RDKit mol from atoms and positions
            rdkit_mol = Chem.RWMol()
            for atom_symbol in atoms:
                atom = Chem.Atom(atom_symbol)
                rdkit_mol.AddAtom(atom)
            
            # Set 3D coordinates
            conf = Chem.Conformer(len(atoms))
            for i, pos in enumerate(positions):
                conf.SetAtomPosition(i, pos)
            rdkit_mol.AddConformer(conf)
            
            # Convert to graph
            graph = self._mol_to_graph(rdkit_mol)
            if graph is not None:
                graphs.append(graph)
                self.successful_indices.append(idx)
                
        # Filter properties if provided
        filtered_properties = None
        if properties is not None and isinstance(properties, (list, np.ndarray)):
            filtered_properties = [properties[idx] for idx in self.successful_indices]
        elif properties is not None and isinstance(properties, (float, int)):
            filtered_properties = [properties] * len(graphs)

        return graphs, filtered_properties

    def _mol_to_graph(self, mol) -> Union[dgl.DGLGraph, None]:
        """Convert RDKit mol to DGL graph following MoleculeDataset pattern"""
        try:
            positions, atom_types, atom_charges, bond_types, bond_idxs, _, _, failed_idxs = self.featurizer.featurize_molecules([mol])
            
            if len(failed_idxs) > 0:
                return None

            # Get first molecule's data
            positions = positions[0].to(torch.float32)
            atom_types = atom_types[0].to(torch.float32)
            atom_charges = atom_charges[0].long()
            bond_types = bond_types[0].to(torch.int64)
            bond_idxs = bond_idxs[0].long()

            # Remove center of mass
            positions = positions - positions.mean(dim=0, keepdim=True)

            # Create adjacency matrix
            n_atoms = positions.shape[0]
            adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int64)
            adj[bond_idxs[:, 0], bond_idxs[:, 1]] = bond_types

            # Get upper triangle and create bidirectional edges
            upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
            upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]
            lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

            edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
            edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

            # One-hot encode edge labels and atom charges
            edge_labels = torch.nn.functional.one_hot(edge_labels.to(torch.int64), num_classes=5).to(torch.float32) # hard-coded assumption of 5 bond types
            atom_charges = torch.nn.functional.one_hot(atom_charges + 2, num_classes=6).to(torch.float32) # hard-coded assumption that charges are in range [-2, 3]

            # Create DGL graph
            g = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)

            # Add features
            g.ndata['x_1_true'] = positions
            g.ndata['a_1_true'] = atom_types
            g.ndata['c_1_true'] = atom_charges
            g.edata['e_1_true'] = edge_labels

            return g
        except Exception as e:
            print(f"Failed to process molecule: {e}")
            return None

    def predict(self, graphs: List[dgl.DGLGraph]) -> torch.Tensor:
        """Make predictions for a list of graphs"""
        if not graphs:  # Check if the list is empty
            print("Warning: No molecules were successfully processed!")
            return None

        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = 128
            for i in range(0, len(graphs), batch_size):
                batch_graphs = graphs[i:i + batch_size]
                batched_graph = collate(batch_graphs)
                batched_graph = batched_graph.to(self.device)
                
                pred = self.model(batched_graph)
                
                # Denormalize predictions if normalization parameters exist
                if self.norm_params is not None:
                    if self.dataset_name == 'qm9':
                        pred = pred * self.norm_params['std'][self.property_idx].to(pred.device) + self.norm_params['mean'][self.property_idx].to(pred.device)
                    elif self.dataset_name == 'qme14s':
                        pred = pred * self.norm_params['std'].to(pred.device) + self.norm_params['mean'].to(pred.device)
                predictions.append(pred.cpu())
        
        return torch.cat(predictions, dim=0) if predictions else None

regressor_folder = "/blue/mingjieliu/jiruijin/github/MolGuidance/molguidance/property_regressor"
# config = f'{regressor_folder}/configs/test.yaml'
config = f'{regressor_folder}/configs/test_qme14s.yaml'

def get_gvp_preds(mols, property_name):
    n_mols = len(mols)
    # checkpoint = f'{regressor_folder}/model_output/{property_name}/checkpoints/last.ckpt'
    checkpoint = f'{regressor_folder}/model_output_qme14s/{property_name}/checkpoints/gvp-regressor-epoch=443-val_loss=0.0334.ckpt'
    model = MoleculePredictor(checkpoint, config, property_name)
    graph_valid = []
    graphs = []
    for idx, mol in enumerate(tqdm(mols, desc="Generating Molecule Graphs")):
        if mol is None:
            continue
        graph = mol_to_graph(mol)
        if graph is not None:
            graphs.append(graph)
            graph_valid.append(idx)
    predictions = model.predict(graphs)
    predictions = predictions.squeeze().numpy()
    _graph_valid = []
    gvp_preds = []
    for g_id, pred in zip(graph_valid, predictions):
        if pred is not None:
            _graph_valid.append(g_id)
            gvp_preds.append(pred)
    full_preds = map_gvp_pred_to_full(gvp_preds, _graph_valid, n_mols)
    return full_preds, _graph_valid

def map_gvp_pred_to_full(gvp_preds, graph_valid, n_mols):
    arr_full = [None] * n_mols
    curr = 0
    for i in range(n_mols):
        if i in graph_valid:
            arr_full[i] = gvp_preds[curr]
            curr += 1
    return arr_full

def mol_to_graph(mol):
        """Convert RDKit mol to DGL graph following MoleculeDataset pattern"""
        with open(config, 'r') as f:
            _config = yaml.safe_load(f)
        featurizer = MoleculeFeaturizer(_config['dataset']['atom_map'])

        try:
            positions, atom_types, atom_charges, bond_types, bond_idxs, _, _, failed_idxs = featurizer.featurize_molecules([mol])

            # Get first molecule's data
            positions = positions[0].to(torch.float32)
            atom_types = atom_types[0].to(torch.float32)
            atom_charges = atom_charges[0].long()
            bond_types = bond_types[0].to(torch.int64)
            bond_idxs = bond_idxs[0].long()

            # Remove center of mass
            positions = positions - positions.mean(dim=0, keepdim=True)

            # Create adjacency matrix
            n_atoms = positions.shape[0]
            adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int64)
            adj[bond_idxs[:, 0], bond_idxs[:, 1]] = bond_types

            # Get upper triangle and create bidirectional edges
            upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
            upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]
            lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

            edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
            edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

            # One-hot encode edge labels and atom charges
            edge_labels = torch.nn.functional.one_hot(edge_labels.to(torch.int64), num_classes=5).to(torch.float32)
            atom_charges = torch.nn.functional.one_hot(atom_charges + 2, num_classes=6).to(torch.float32)

            # Create DGL graph
            g = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)

            # Add features
            g.ndata['x_1_true'] = positions
            g.ndata['a_1_true'] = atom_types
            g.ndata['c_1_true'] = atom_charges
            g.edata['e_1_true'] = edge_labels

            return g
        except Exception as e:
            print(f"Failed to process molecule: {e}")
            return None

def compute_mae_from_sdf(sdf_file, property_name, target_properties):
    mols = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    gvp_preds, _ = get_gvp_preds(mols, property_name)
    mae = calculate_mae(target_properties, gvp_preds)
    return mae

def save_mol_to_sdf(mols, filename="rdkit_mols.sdf"):
    writer = Chem.SDWriter(filename)
    writer.SetKekulize(False)
    for mol in mols:
        writer.write(mol)
    writer.close()
    print(f"✅ Molecule saved to {filename}")

def calculate_mae(true_values, predicted_values):
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    
    # Ensure the two arrays have the same length
    if true_values.shape != predicted_values.shape:
        raise ValueError("Both input arrays must have the same shape. " f"{true_values.shape} != {predicted_values.shape}")
    
    mae = np.mean(np.abs(true_values - predicted_values))
    return mae