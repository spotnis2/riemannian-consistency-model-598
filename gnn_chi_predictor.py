"""
gnn_chi_predictor.py
--------------------
GNN that directly predicts chi angles from backbone structure.
No flow matching, no distillation — pure supervised learning.

Architecture:
  - Node features: aa_onehot(21) + bb_dihedral sin/cos(6) = 27 dims
  - Edge features: distance(1) + relative position sin/cos(6) = 7 dims
  - 6 rounds of message passing on kNN graph (k=16)
  - Output: 4 chi angles per residue (as sin/cos pairs = 8 dims, decoded to angles)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

NODE_IN_DIM = 27   # aa_onehot(21) + bb_dihedral sin/cos(6)
EDGE_IN_DIM = 7    # distance(1) + rel_pos sin/cos(6)
HIDDEN_DIM  = 256
N_LAYERS    = 6
K_NEIGHBORS = 16


# Added dropout for regularization
# Added dropout for regularization
class EdgeLayer(nn.Module):
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
        return self.mlp(torch.cat([h[src], h[dst], edge_attr], dim=-1))


class NodeLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, h, messages, edge_index, n_nodes):
        src, dst = edge_index
        agg = scatter_mean(messages, dst, dim=0, dim_size=n_nodes)
        return self.norm(h + self.mlp(torch.cat([h, agg], dim=-1)))


class GNNChiPredictor(nn.Module):
    """
    GNN that predicts chi angles directly from backbone structure.
    Outputs sin/cos of each chi angle for numerical stability.
    """
    def __init__(self, node_in=NODE_IN_DIM, edge_in=EDGE_IN_DIM,
                 hidden=HIDDEN_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_embed = nn.Linear(edge_in, edge_in)

        self.edge_layers = nn.ModuleList([EdgeLayer(hidden, edge_in, hidden) for _ in range(n_layers)])
        self.node_layers = nn.ModuleList([NodeLayer(hidden) for _ in range(n_layers)])

        # Output: sin and cos of each of 4 chi angles = 8 values
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 8),  # 4 sin + 4 cos
        )

    def forward(self, node_feat, edge_index, edge_attr):
        N = node_feat.shape[0]
        h = self.node_embed(node_feat)
        e = self.edge_embed(edge_attr)

        for el, nl in zip(self.edge_layers, self.node_layers):
            msg = el(h, edge_index, e)
            h   = nl(h, msg, edge_index, N)

        out = self.output_mlp(h)  # (N, 8)
        sin_pred = out[:, :4]     # (N, 4)
        cos_pred = out[:, 4:]     # (N, 4)

        # Normalize to unit circle
        norm = torch.sqrt(sin_pred**2 + cos_pred**2 + 1e-8)
        sin_pred = sin_pred / norm
        cos_pred = cos_pred / norm

        # Convert to angles in [0, 2pi]
        angles = torch.atan2(sin_pred, cos_pred) % (2 * 3.14159265358979)
        return angles  # (N, 4)


def build_graph(bb, bb_mask, aa_onehot, bb_dihedral, k=K_NEIGHBORS, device='cuda'):
    """
    Build kNN graph for valid residues.

    Args:
        bb:          (512, 4, 3) backbone coords
        bb_mask:     (512,) valid residue mask
        aa_onehot:   (512, 21)
        bb_dihedral: (512, 3)

    Returns:
        node_feat:  (N_valid, 27)
        edge_index: (2, E)
        edge_attr:  (E, 7)
        valid:      (512,) bool mask
    """
    valid = bb_mask.bool()
    ca    = bb[valid, 1, :]   # (N_valid, 3)
    N     = ca.shape[0]

    if N < 2:
        return None, None, None, valid

    # Node features: aa + bb_dihedral sin/cos — all rotation invariant
    node_feat = torch.cat([
        aa_onehot[valid],           # (N, 21)
        bb_dihedral[valid].sin(),   # (N, 3)
        bb_dihedral[valid].cos(),   # (N, 3)
    ], dim=-1)  # (N, 27)

    # kNN edges
    k_actual = min(k, N-1)
    dist_mat = torch.cdist(ca, ca)
    dist_mat.fill_diagonal_(float('inf'))
    _, knn_idx = dist_mat.topk(k_actual, largest=False, dim=-1)

    src = torch.arange(N, device=device).unsqueeze(-1).expand(-1, k_actual).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    # Edge features: distance + relative direction
    diff     = ca[dst] - ca[src]
    dist     = diff.norm(dim=-1, keepdim=True)
    dist_norm = dist / (dist.max() + 1e-8)
    pos_enc  = torch.cat([diff.sin(), diff.cos()], dim=-1)  # (E, 6)
    edge_attr = torch.cat([dist_norm, pos_enc], dim=-1)     # (E, 7)

    return node_feat, edge_index, edge_attr, valid