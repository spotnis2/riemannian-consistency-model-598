"""
CachedVFDataset: loads precomputed FlowPacker vector fields from disk.
Uses 27-dim rotation-invariant per-residue conditioning:
  - aa_onehot (21)
  - bb_dihedral sin (3)
  - bb_dihedral cos (3)
  Total: 27 dims (no raw Cartesian coords)
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'X':'X','UNK':'X',
}
ONE_LETTER = 'ACDEFGHIKLMNPQRSTVWYX'
LETTER_TO_NUM = {c: i for i, c in enumerate(ONE_LETTER)}

COND_DIM = 27  # aa_onehot(21) + bb_dihedral sin/cos(6)


def _get_bb_dihedral(n, ca, c):
    L = ca.shape[0]
    dihedrals = torch.zeros(L, 3)

    def dihedral(a, b, cc, d):
        b1=b-a; b2=cc-b; b3=d-cc
        n1=torch.cross(b1,b2,dim=-1); n2=torch.cross(b2,b3,dim=-1)
        m1=torch.cross(n1,b2/(b2.norm(dim=-1,keepdim=True)+1e-8),dim=-1)
        return torch.atan2((m1*n2).sum(-1),(n1*n2).sum(-1))

    if L > 1:
        dihedrals[1:,0]  = dihedral(c[:-1],n[1:],ca[1:],c[1:])
        dihedrals[:-1,1] = dihedral(n[:-1],ca[:-1],c[:-1],n[1:])
        dihedrals[1:,2]  = dihedral(ca[:-1],c[:-1],n[1:],ca[1:])
    return dihedrals


class CachedVFDataset(Dataset):
    """
    Returns (xt, vf, t, chi_mask, cond) per protein.
    cond is 27-dim rotation-invariant per-residue features.
    """

    def __init__(self,
                 cache_dir='datasets/vf_cache2',
                 side_chain_path='datasets/side_chain_data.npz',
                 sequence_path='datasets/sequence_data.npz',
                 backbone_path='datasets/backbone_data.npz',
                 **kwargs):

        self.cache_dir = cache_dir
        self.cache_files = sorted([
            f for f in os.listdir(cache_dir) if f.endswith('.pt')
        ])
        self.N = len(self.cache_files)
        print(f'CachedVFDataset: {self.N} proteins, cond_dim={COND_DIM}')

        sc = np.load(side_chain_path)
        self.chi_mask = torch.tensor(sc['mask'], dtype=torch.float32)  # (N,512,4)

        seq = np.load(sequence_path, allow_pickle=True)
        self.names = seq['names']  # (N,512)

        bb = np.load(backbone_path)
        self.backbone = torch.tensor(bb['backbone'], dtype=torch.float32)  # (N,512,4,3)
        self.bb_mask  = torch.tensor(bb['mask'],     dtype=torch.float32)  # (N,512)

    def __len__(self):
        return self.N

    @property
    def dimension(self):
        return 4

    def __getitem__(self, idx):
        cache = torch.load(
            os.path.join(self.cache_dir, self.cache_files[idx]),
            map_location='cpu'
        )
        vf_all = cache['vf']   # (N_T,512,4)
        t_all  = cache['t']    # (N_T,512,1)
        xt_all = cache['xt']   # (N_T,512,4)
        N_T = vf_all.shape[0]

        k = torch.randint(0, N_T, (1,)).item()
        vf_k  = vf_all[k]        # (512,4)
        t_k   = t_all[k,:,0]     # (512,)
        xt_k  = xt_all[k]        # (512,4)

        chi_mask = self.chi_mask[idx]   # (512,4)
        bb       = self.backbone[idx]   # (512,4,3)
        bb_mask  = self.bb_mask[idx]    # (512,)
        names    = self.names[idx]      # (512,)

        aa_num = torch.tensor(
            [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in names],
            dtype=torch.long
        )
        aa_onehot = F.one_hot(aa_num, num_classes=21).float()  # (512,21)

        bb_dihedral = _get_bb_dihedral(bb[:,0], bb[:,1], bb[:,2])  # (512,3)

        # 27-dim invariant cond: no raw Cartesian coordinates
        cond = torch.cat([
            aa_onehot,           # (512,21) — amino acid identity
            bb_dihedral.sin(),   # (512,3)  — invariant backbone geometry
            bb_dihedral.cos(),   # (512,3)  — invariant backbone geometry
        ], dim=-1) * bb_mask.unsqueeze(-1)  # (512,27)

        return {
            'xt':       xt_k,      # (512,4)
            'vf':       vf_k,      # (512,4)
            't':        t_k,       # (512,)
            'chi_mask': chi_mask,  # (512,4)
            'cond':     cond,      # (512,27)
        }