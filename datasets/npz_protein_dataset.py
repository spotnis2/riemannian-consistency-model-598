"""
NpzProteinDataset: builds FlowPacker-compatible PyG Data objects from
preprocessed .npz files (backbone_data.npz, side_chain_data.npz, sequence_data.npz).

No PDB loading needed — uses the same preprocessed data as the RCT pipeline.
"""

import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph
import warnings

# FlowPacker constants — must match flowpacker/utils/constants.py
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'X': 'X', 'UNK': 'X',
}
ONE_LETTER = 'ACDEFGHIKLMNPQRSTVWYX'
LETTER_TO_NUM = {c: i for i, c in enumerate(ONE_LETTER)}


def get_bb_dihedral(n, ca, c):
    """Compute backbone phi/psi/omega dihedrals. Returns (L, 3)."""
    L = ca.shape[0]
    dihedrals = torch.zeros(L, 3, device=ca.device)

    def dihedral(a, b, c, d):
        b1 = b - a
        b2 = c - b
        b3 = d - c
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        m1 = torch.cross(n1, b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8), dim=-1)
        x = (n1 * n2).sum(-1)
        y = (m1 * n2).sum(-1)
        return torch.atan2(y, x)

    # omega: C(i-1)-N(i)-CA(i)-C(i)  — skip first residue
    # phi:   C(i-1)-N(i)-CA(i)-C(i)  — skip first residue
    # psi:   N(i)-CA(i)-C(i)-N(i+1)  — skip last residue
    if L > 1:
        dihedrals[1:, 0] = dihedral(c[:-1], n[1:], ca[1:], c[1:])   # phi
        dihedrals[:-1, 1] = dihedral(n[:-1], ca[:-1], c[:-1], n[1:]) # psi
        dihedrals[1:, 2] = dihedral(ca[:-1], c[:-1], n[1:], ca[1:])  # omega
    return dihedrals


def filter_edges_by_residue_mask(edge_index, residue_mask):
    if edge_index.numel() == 0:
        return edge_index
    row, col = edge_index
    keep = (residue_mask[row] != 0) & (residue_mask[col] != 0)
    return edge_index[:, keep]


class NpzProteinDataset(Dataset):
    """
    Loads preprocessed protein data from .npz files and returns
    PyG Data objects compatible with FlowPacker's get_vf().

    Required files:
      - backbone_data.npz:  'backbone' (N, 512, 4, 3), 'mask' (N, 512)
      - side_chain_data.npz: 'side_chains' (N, 512, 4), 'mask' (N, 512, 4)
      - sequence_data.npz:  'names' (N, 512) [3-letter AA codes], 'mask' (N, 512)
    """

    def __init__(self,
                 backbone_path='datasets/backbone_data.npz',
                 side_chain_path='datasets/side_chain_data.npz',
                 sequence_path='datasets/sequence_data.npz',
                 max_radius=8.0,
                 max_num_neighbors=30,
                 edge_type='radius',
                 **kwargs):

        self.max_radius = max_radius
        self.max_num_neighbors = max_num_neighbors
        self.edge_type = edge_type

        bb = np.load(backbone_path)
        sc = np.load(side_chain_path)
        seq = np.load(sequence_path, allow_pickle=True)

        self.backbone = bb['backbone'].astype(np.float32)    # (N, 512, 4, 3)
        self.bb_mask = bb['mask'].astype(np.float32)         # (N, 512)
        self.chi = sc['side_chains'].astype(np.float32)      # (N, 512, 4)
        self.chi_mask = sc['mask'].astype(np.float32)        # (N, 512, 4)
        self.names = seq['names']                            # (N, 512) 3-letter str

        self.N = self.backbone.shape[0]
        print(f'NpzProteinDataset: {self.N} proteins loaded')

    def __len__(self):
        return self.N

    @property
    def dimension(self):
        return 4

    def __getitem__(self, idx):
        bb = torch.tensor(self.backbone[idx])       # (512, 4, 3)
        mask = torch.tensor(self.bb_mask[idx])      # (512,)
        chi = torch.tensor(self.chi[idx])           # (512, 4)
        chi_mask = torch.tensor(self.chi_mask[idx]) # (512, 4)
        names = self.names[idx]                     # (512,) str

        # Amino acid encoding
        aa_num = torch.tensor(
            [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n), 'X'), 20) for n in names],
            dtype=torch.long
        )  # (512,)
        aa_onehot = F.one_hot(aa_num, num_classes=21).float()  # (512, 21)

        # pos: (512, 4, 3) — N, CA, C, O backbone atoms
        pos = bb  # already (512, 4, 3)

        # Backbone dihedrals from N, CA, C atoms
        n_atoms = bb[:, 0, :]   # (512, 3)
        ca_atoms = bb[:, 1, :]  # (512, 3)
        c_atoms  = bb[:, 2, :]  # (512, 3)
        bb_dihedral = get_bb_dihedral(n_atoms, ca_atoms, c_atoms)  # (512, 3)

        # atom_mask: (512, 14) — FlowPacker uses 14 heavy atoms
        # We only have backbone (4 atoms), so mark first 4 as present where mask=1
        atom_mask = torch.zeros(512, 14, dtype=torch.long)
        atom_mask[:, :4] = mask.unsqueeze(-1).long()

        # chi angles: shift from [0,2pi) to [-pi,pi] for FlowPacker
        chi_fp = (chi - math.pi) * chi_mask  # FlowPacker expects [-pi,pi]*mask

        # Edge index on CA coordinates
        ca = ca_atoms  # (512, 3)
        # Only use valid residues for edges
        valid_mask = mask.bool()

        if self.edge_type == 'radius':
            edge_index = radius_graph(ca, r=self.max_radius,
                                      max_num_neighbors=self.max_num_neighbors)
        else:
            edge_index = knn_graph(ca, k=self.max_num_neighbors)

        edge_index = filter_edges_by_residue_mask(edge_index, mask)

        # chi_alt_mask: symmetric chi angles (e.g. PHE, TYR) — zeros for simplicity
        chi_alt_mask = torch.zeros(512, dtype=torch.bool)

        data = Data(
            pos=pos,                    # (512, 4, 3)
            bb_dihedral=bb_dihedral,    # (512, 3)
            aa_onehot=aa_onehot,        # (512, 21)
            aa_num=aa_num,              # (512,)
            aa_mask=mask,               # (512,) float
            chi=chi_fp,                 # (512, 4) in [-pi,pi]*mask
            chi_mask=chi_mask,          # (512, 4)
            chi_alt_mask=chi_alt_mask,  # (512,)
            atom_mask=atom_mask,        # (512, 14)
            edge_index=edge_index,
        )
        return data