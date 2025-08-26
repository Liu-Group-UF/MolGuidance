import numpy as np
import torch
from torch.distributions.categorical import Categorical
from typing import Dict, Tuple, Union, Optional
from torchtyping import TensorType
from typeguard import typechecked
from pathlib import Path
import pickle

class NumNodesDistribution:
    def __init__(
        self,
        histogram: Dict[int, int],
        eps: float = 1e-30
    ):
        self.eps = eps
        num_nodes, self.keys, prob = [], {}, []
        for i, nodes in enumerate(histogram):
            num_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
            
        self.num_nodes = torch.tensor(num_nodes)
        self.prob = torch.tensor(prob)
        self.prob = self.prob / torch.sum(self.prob)
        self.m = Categorical(self.prob)

    @typechecked
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        idx = self.m.sample((n_samples,))
        return self.num_nodes[idx]

    @typechecked
    def log_prob(self, batch_n_nodes: TensorType["batch_size"]) -> TensorType["batch_size"]:
        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs, device=batch_n_nodes.device)

        log_p = torch.log(self.prob + self.eps)
        log_probs = log_p[idcs]

        return log_probs
    
class SimplePropertySampler:
    """
    Heavily adapted from: https://github.com/ehoogeboom/e3_diffusion_for_molecules 
    and https://github.com/BioinfoMachineLearning/bio-diffusion/
    blob/main/src/models/__init__.py
    """
    def __init__(
        self,
        n_nodes: Union[np.ndarray, torch.Tensor],
        property_values: Union[np.ndarray, torch.Tensor],
        num_bins: int = 1000,
        device: str = "cpu",
        seed: Optional[int] = None
    ):
        """Initialize property sampler with node counts and corresponding property values.
        
        Args:
            n_nodes: Array of node counts [num_examples]
            property_values: Array of property values [num_examples]
            num_bins: Number of bins for discretizing property distribution
            device: Device to store tensors on
        """
        self.device = device
        self.num_bins = num_bins
        self.distributions = {}
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        # Convert numpy arrays to tensors if needed
        n_nodes = torch.as_tensor(n_nodes, device=device)
        property_values = torch.as_tensor(property_values, device=device)
        
        # Store node distribution for sampling
        self.nodes_histogram = {}
        unique_nodes, counts = torch.unique(n_nodes, return_counts=True)
        for n, c in zip(unique_nodes, counts):
            self.nodes_histogram[n.item()] = c.item()
        
        # Create node distribution
        self.node_dist = NumNodesDistribution(self.nodes_histogram)
        
        # Create probability distribution for each unique node count
        self._create_distributions(n_nodes, property_values)
        
    @typechecked    
    def save(self, filepath: str):
        """Save sampler state to file.
        
        Args:
            filepath: Path to save the sampler state
        """
        save_dict = {
            'nodes_histogram': self.nodes_histogram,
            'distributions': self.distributions,
            'num_bins': self.num_bins,
            'device': self.device
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
            
    @classmethod
    @typechecked
    def load(cls, filepath: str, device: str = "cpu") -> "SimplePropertySampler":
        """Load sampler state from file.
        
        Args:
            filepath: Path to load the sampler state from
            device: Device to load the sampler onto
            
        Returns:
            Loaded sampler instance
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
            
        # Create dummy sampler with minimal data
        dummy_nodes = np.array([1])
        dummy_props = np.array([0.0])
        sampler = cls(dummy_nodes, dummy_props, num_bins=save_dict['num_bins'], device=device)
        
        # Restore saved state
        sampler.nodes_histogram = save_dict['nodes_histogram']
        sampler.distributions = save_dict['distributions']
        sampler.node_dist = NumNodesDistribution(sampler.nodes_histogram)
        return sampler
    
    @typechecked
    def _create_distributions(
        self,
        n_nodes: TensorType["num_examples"],
        values: TensorType["num_examples"]
    ):
        min_nodes = int(torch.min(n_nodes))
        max_nodes = int(torch.max(n_nodes))
        
        for n in range(min_nodes, max_nodes + 1):
            idxs = n_nodes == n
            values_filtered = values[idxs]
            
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                self.distributions[n] = {
                    "probs": probs,
                    "params": params
                }

    @typechecked
    def _create_prob_given_nodes(
        self,
        values: TensorType["num_matched_values"]
    ) -> Tuple[Categorical, Tuple[torch.Tensor, torch.Tensor]]:
        prop_min = torch.min(values)
        prop_max = torch.max(values)
        prop_range = prop_max - prop_min + 1e-12

        # Create histogram
        histogram = torch.zeros(self.num_bins, device=self.device)
        for val in values:
            i = int((val - prop_min)/prop_range * self.num_bins)
            if i == self.num_bins:
                i = self.num_bins - 1
            histogram[i] += 1
            
        # Create categorical distribution
        probs = Categorical(histogram / torch.sum(histogram))
        params = (prop_min, prop_max)
        return probs, params

    @typechecked
    def _idx2value(
        self,
        idx: torch.Tensor,
        params: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Convert bin index to continuous value."""
        prop_min, prop_max = params
        prop_range = prop_max - prop_min
        left = idx / self.num_bins * prop_range + prop_min
        right = (idx + 1) / self.num_bins * prop_range + prop_min
        val = torch.rand(1, device=self.device) * (right - left) + left
        return val

    @typechecked    
    def sample(self, n_samples: int, seed=None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample property values and corresponding node counts.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of:
                - Array of property values [n_samples]
                - Array of node counts [n_samples]
        """
        # Set temporary seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        # Sample node counts
        sampled_nodes = self.node_dist.sample(n_samples)
        
        # Sample properties for each node count
        sampled_props = []
        for n_nodes in sampled_nodes:
            if n_nodes.item() not in self.distributions:
                raise ValueError(f"No distribution available for {n_nodes.item()} nodes")
                
            dist = self.distributions[n_nodes.item()]
            # Sample bin index
            idx = dist["probs"].sample((1,))
            # Convert to value
            val = self._idx2value(idx, dist["params"])
            sampled_props.append(val)
            
        # Convert to numpy arrays
        sampled_props = torch.cat(sampled_props).cpu().numpy()
        sampled_nodes = sampled_nodes.cpu().numpy()
        
        return sampled_props, sampled_nodes

def sample_n_atoms(property_values, n_atoms_list, property_range, n_samples=1, random_seed=None):
    """
    Sample n_atoms corresponding to property values within a specified range around the target value.
    
    Args:
        property_values (list or np.array): List of property values.
        n_atoms_list (list or np.array): List of number of atoms, same length as property_values.
        target_value (float): Target property value to sample around.
        tolerance (float): Acceptable deviation from the target_value (default 0.01).
        n_samples (int): Number of samples to draw (default 1).
        random_seed (int or None): Random seed for reproducibility.

    Returns:
        sampled_n_atoms (list): List of sampled number of atoms.
    """
    property_values = np.array(property_values)
    n_atoms_list = np.array(n_atoms_list)
    
    low_property, high_property = min(property_range), max(property_range)
    # Find indices where property is within the desired range
    mask = (property_values >= low_property) & (property_values <= high_property) 
    matching_n_atoms = n_atoms_list[mask]
    if len(matching_n_atoms) == 0:
        raise ValueError(f"No matching property values found within {tolerance} of {property_range}.")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    sampled_n_atoms = np.random.choice(matching_n_atoms, size=n_samples, replace=True)
    
    return sampled_n_atoms.tolist()