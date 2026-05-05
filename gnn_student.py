"""
gnn_student.py
--------------
Small graph neural network student for RCD distillation from FlowPacker.

Architecture:
  - Node features: aa_onehot(21) + bb_dihedral sin/cos(6) + xt(4) + t(1) = 32 dims
  - Edge features: relative position encoding + distance (7 dims)
  - 4 rounds of message passing on kNN graph (k=16)
  - Output: 4-dim vector field per residue

Much lighter than FlowPacker's EquiformerV2 but captures local structural context
through neighborhood aggregation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


NODE_IN_DIM  = 32   # aa(21) + bb_dihedral sin/cos(6) + xt(4) + t(1)
EDGE_IN_DIM  = 7    # distance(1) + relative_position_encoding(6)
HIDDEN_DIM   = 128
N_LAYERS     = 4
K_NEIGHBORS  = 16
OUT_DIM      = 4


class EdgeLayer(nn.Module):
    """Compute edge messages from source, dest, and edge features."""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        msg_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.mlp(msg_input)  # (E, hidden_dim)


class NodeLayer(nn.Module):
    """Update node features by aggregating messages."""
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, messages, edge_index, n_nodes):
        src, dst = edge_index
        # Aggregate messages to destination nodes
        agg = scatter_mean(messages, dst, dim=0, dim_size=n_nodes)  # (N, hidden_dim)
        h_new = self.mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h_new + h if h.shape[-1] == h_new.shape[-1] else h_new)


class GNNStudent(nn.Module):
    """
    Small GNN for predicting FlowPacker's vector field.
    Operates on per-residue graph with kNN edges based on CA distances.
    """

    def __init__(self,
                 node_in_dim=NODE_IN_DIM,
                 edge_in_dim=EDGE_IN_DIM,
                 hidden_dim=HIDDEN_DIM,
                 n_layers=N_LAYERS,
                 out_dim=OUT_DIM):
        super().__init__()

        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed  = nn.Linear(edge_in_dim, edge_in_dim)  # pass-through with learned transform

        self.edge_layers = nn.ModuleList([
            EdgeLayer(hidden_dim, edge_in_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.node_layers = nn.ModuleList([
            NodeLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, node_feat, edge_index, edge_attr):
        """
        Args:
            node_feat:  (N, node_in_dim)
            edge_index: (2, E) — src, dst pairs
            edge_attr:  (E, edge_in_dim)
        Returns:
            vf: (N, out_dim)
        """
        N = node_feat.shape[0]
        h = self.node_embed(node_feat)  # (N, hidden_dim)
        e = self.edge_embed(edge_attr)  # (E, edge_in_dim)

        for edge_layer, node_layer in zip(self.edge_layers, self.node_layers):
            msg = edge_layer(h, edge_index, e)       # (E, hidden_dim)
            h   = node_layer(h, msg, edge_index, N)  # (N, hidden_dim)

        return self.output_mlp(h)  # (N, out_dim)


def build_graph(bb, bb_mask, xt, t_scalar, aa_onehot, bb_dihedral, k=K_NEIGHBORS, device='cuda'):
    """
    Build per-protein graph for GNN.

    Args:
        bb:          (512, 4, 3) backbone coordinates
        bb_mask:     (512,) valid residue mask
        xt:          (512, 4) noised chi angles
        t_scalar:    (512,) timestep per residue
        aa_onehot:   (512, 21)
        bb_dihedral: (512, 3)

    Returns:
        node_feat:  (N_valid, 32)
        edge_index: (2, E)
        edge_attr:  (E, 7)
        valid_mask: (512,) bool — which residues are valid
    """
    valid = bb_mask.bool()  # (512,)
    ca = bb[valid, 1, :]    # (N_valid, 3) CA coordinates

    N = ca.shape[0]
    if N == 0:
        return None, None, None, valid

    # Node features: aa(21) + bb_dihedral sin/cos(6) + xt(4) + t(1)
    node_feat = torch.cat([
        aa_onehot[valid],              # (N, 21)
        bb_dihedral[valid].sin(),      # (N, 3)
        bb_dihedral[valid].cos(),      # (N, 3)
        xt[valid],                     # (N, 4)
        t_scalar[valid].unsqueeze(-1), # (N, 1)
    ], dim=-1)  # (N, 32)

    # Build kNN edges
    k_actual = min(k, N-1)
    if k_actual == 0:
        # Single residue — no edges, use self-loop
        edge_index = torch.zeros(2, 1, dtype=torch.long, device=device)
        edge_attr  = torch.zeros(1, EDGE_IN_DIM, device=device)
        return node_feat, edge_index, edge_attr, valid

    # Pairwise distances
    dist_mat = torch.cdist(ca, ca)  # (N, N)
    dist_mat.fill_diagonal_(float('inf'))

    # kNN
    _, knn_idx = dist_mat.topk(k_actual, largest=False, dim=-1)  # (N, k)
    src = torch.arange(N, device=device).unsqueeze(-1).expand(-1, k_actual).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)  # (2, N*k)

    # Edge features: distance + relative position (projected to 6 dims)
    diff = ca[dst] - ca[src]                    # (E, 3)
    dist = diff.norm(dim=-1, keepdim=True)      # (E, 1)
    dist_norm = dist / (dist.max() + 1e-8)

    # Encode relative position with sin/cos in 3 axes = 6 dims
    pos_enc = torch.cat([diff.sin(), diff.cos()], dim=-1)  # (E, 6)
    edge_attr = torch.cat([dist_norm, pos_enc], dim=-1)    # (E, 7)

    return node_feat, edge_index, edge_attr, valid