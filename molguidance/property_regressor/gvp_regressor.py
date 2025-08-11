from molguidance.models.gvp import GVPConv, _norm_no_nan, GVP, _rbf
from molguidance.models.vector_field import NodePositionUpdate, EdgeUpdate
from typing import Union, List
import torch.nn as nn
import torch
import dgl
import dgl.function as fn

class GVPRegressor(nn.Module):
    """
    GVP regressor class for predicting molecular properties.
    Incorporates position and edge feature updates during message passing.
    """
    def __init__(self, 
                 scalar_size: int = 128,
                 vector_size: int = 16,
                 n_cp_feats: int = 0,
                 n_message_gvps: int = 1,
                 n_update_gvps: int = 1,
                 edge_feat_size: int = 64,
                 rbf_dmax: float = 20,
                 rbf_dim: int = 16,
                 message_norm: Union[float, str] = 100,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 n_tasks: int = 1,
                 pooling_type: str = 'mean',
                 convs_per_update: int = 2,
                 update_positions: bool = True,
                 n_atom_types: int = 5,
                ):
        super(GVPRegressor, self).__init__()
        
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_tasks = n_tasks
        self.pooling_type = pooling_type
        self.convs_per_update = convs_per_update
        self.update_positions = update_positions
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        # Create GVP convolution layers
        self.conv_layers = nn.ModuleList([
            GVPConv(
                scalar_size=scalar_size,
                vector_size=vector_size,
                n_cp_feats=n_cp_feats,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                edge_feat_size=edge_feat_size,
                rbf_dmax=rbf_dmax,
                rbf_dim=rbf_dim,
                message_norm=message_norm,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        if update_positions:
            # Position and edge updaters
            self.node_position_updaters = nn.ModuleList([
                NodePositionUpdate(scalar_size, vector_size, n_gvps=3, n_cp_feats=n_cp_feats)
                for _ in range((num_layers + convs_per_update - 1) // convs_per_update)
            ])
            
            self.edge_updaters = nn.ModuleList([
                EdgeUpdate(scalar_size, edge_feat_size, update_edge_w_distance=True, rbf_dim=rbf_dim)
                for _ in range((num_layers + convs_per_update - 1) // convs_per_update)
            ])

        # Node feature embeddings
        self.scalar_embedding = nn.Sequential(
            nn.Linear(n_atom_types, scalar_size),
            nn.SiLU(),
            nn.Linear(scalar_size, scalar_size),
            nn.SiLU(),
            nn.LayerNorm(scalar_size)
        )

        # Edge feature embeddings
        n_bond_types = 5  # From dataset.py
        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types, edge_feat_size),
            nn.SiLU(),
            nn.Linear(edge_feat_size, edge_feat_size),
            nn.SiLU(),
            nn.LayerNorm(edge_feat_size)
        )

        # Optional attention-based pooling
        if pooling_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(scalar_size, scalar_size),
                nn.SiLU(),
                nn.Linear(scalar_size, 1)
            )

        # Final prediction layers
        self.graph_predictor = nn.Sequential(
            nn.Linear(scalar_size, scalar_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(scalar_size, n_tasks)
        )

    def precompute_distances(self, g: dgl.DGLGraph, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""
        with g.local_scope():
            if node_positions is None:
                g.ndata['x_d'] = g.ndata['x_1_true']
            else:
                g.ndata['x_d'] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return x_diff, d

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Forward pass of the GVP regressor."""
        # Get initial node and edge features
        atom_types = g.ndata['a_1_true'].float()
        # charge_features = g.ndata['c_1_true'].float() # we tested without charge features model performance same as with charge features
        positions = g.ndata['x_1_true']
        edge_features = g.edata['e_1_true']

        # Embed initial features
        scalar_feats = self.scalar_embedding(atom_types)
        # scalar_feats = self.scalar_embedding(torch.cat([atom_types, charge_features], dim=-1))
        edge_feats = self.edge_embedding(edge_features)
        
        # Initialize vector features
        num_nodes = g.num_nodes()
        vector_feats = torch.zeros((num_nodes, self.vector_size, 3), device=g.device)
        
        # Process through GVP layers with optional position and edge updates
        x_diff, d = self.precompute_distances(g, positions)
        
        for conv_idx, conv in enumerate(self.conv_layers):
            # Perform convolution
            scalar_feats, vector_feats = conv(
                g,
                scalar_feats=scalar_feats,
                coord_feats=positions,
                vec_feats=vector_feats,
                edge_feats=edge_feats,
                x_diff=x_diff,
                d=d
            )
            
            # Update positions and edge features if enabled
            if self.update_positions and conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:
                updater_idx = conv_idx // self.convs_per_update
                positions = self.node_position_updaters[updater_idx](scalar_feats, positions, vector_feats)
                x_diff, d = self.precompute_distances(g, positions)
                edge_feats = self.edge_updaters[updater_idx](g, scalar_feats, edge_feats, d=d)
            
        # Store final node representations
        g.ndata['h'] = scalar_feats
        
        # Perform graph pooling based on specified method
        if self.pooling_type == 'attention':
            # Compute attention weights
            attention_weights = torch.sigmoid(self.attention(scalar_feats))
            g.ndata['attn'] = attention_weights
            pooled = dgl.readout_nodes(g, 'h', weight='attn')
        elif self.pooling_type == 'sum':
            pooled = dgl.readout_nodes(g, 'h', op='sum')
        else:  # default to mean pooling
            pooled = dgl.readout_nodes(g, 'h', op='mean')
        
        # Predict properties
        predictions = self.graph_predictor(pooled)
        return predictions

