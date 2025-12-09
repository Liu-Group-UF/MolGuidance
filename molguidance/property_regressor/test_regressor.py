import argparse
from pathlib import Path
import torch
import yaml
from rdkit import Chem
# import ase.io  # only install ase if need to use xyz file as input
import dgl
from typing import List, Union
from tqdm import tqdm
import numpy as np
import os

from molguidance.data_processing.geom import MoleculeFeaturizer
from molguidance.property_regressor.train_regressor import GVPRegressorModule
from molguidance.data_processing.dataset import collate
from molguidance.models.ctmc_vector_field import PROPERTY_MAP


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
        print(f"Loading normalization parameters from: {norm_params_path}")
        self.dataset_name = self.config['dataset'].get('dataset_name', 'qm9')
        print(f"Dataset name: {self.dataset_name}")
        if self.dataset_name == 'qm9':
            self.property_idx = PROPERTY_MAP[property_name]
        if norm_params_path.exists():
            self.norm_params = torch.load(norm_params_path)
            print(f"std: {self.norm_params['std']}, mean: {self.norm_params['mean']}")
        else:
            self.norm_params = None

    def process_sdf(self, sdf_path: str, properties=None) -> tuple:
        """Process molecules from an SDF file"""
        mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
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

def calculate_mae(true_values, predicted_values):
    """
    Calculate the Mean Absolute Error (MAE) between two lists or arrays.
    
    Parameters:
    - true_values: list or numpy array of actual values
    - predicted_values: list or numpy array of predicted values
    
    Returns:
    - mae: Mean Absolute Error
    """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    
    # Ensure the two arrays have the same length
    if true_values.shape != predicted_values.shape:
        raise ValueError("Both input arrays must have the same shape.")
    
    mae = np.mean(np.abs(true_values - predicted_values))
    return mae

def str_or_float(value):
    try:
        return float(value)
    except ValueError:
        return value  # return as string if it can't be converted to float

def main():
    parser = argparse.ArgumentParser(description='Predict properties for molecules')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Path to input file (SDF) or directory of XYZ files')
    parser.add_argument('--output', type=str, required=True, help='Name of output file')
    parser.add_argument('--properties_values', type=str_or_float, default=None, help='Either a path to a numpy file containing property values or a single value to use for all predictions')
    parser.add_argument('--property_name', type=str, default='mu', help='Name of property to predict')
    args = parser.parse_args()

    # Initialize predictor
    predictor = MoleculePredictor(args.checkpoint, args.config, args.property_name)

    # Load properties if provided
    properties = None
    if args.properties_values and isinstance(args.properties_values, str):
        properties = np.load(args.properties_values).tolist()
    elif args.properties_values and isinstance(args.properties_values, (float, int)):
        properties = args.properties_values        

    # Process input files
    input_path = Path(args.input)
    if input_path.is_file() and input_path.suffix.lower() == '.sdf':
        graphs, filtered_properties = predictor.process_sdf(str(input_path), properties)
    elif input_path.is_dir():
        xyz_files = list(input_path.glob('*.xyz'))
        graphs, filtered_properties = predictor.process_xyz_files([str(f) for f in xyz_files], properties)
    else:
        raise ValueError("Input must be an SDF file or directory containing XYZ files")

    # Make predictions
    predictions = predictor.predict(graphs)
    predictions = predictions.squeeze()

    # Save results
    results = {
        'predictions': predictions.numpy(),
        'property_name': args.property_name, 
    }
    # os.makedirs("prediction_result", exist_ok=True)
    # output_path = Path("prediction_result") / args.output
    # torch.save(results, output_path)
    torch.save(results, args.output)

    if filtered_properties is not None:
        mae = calculate_mae(filtered_properties, results['predictions'])
        print(f"Mean Absolute Error: {mae}")

if __name__ == "__main__":
    main()