"""
train_rcd_gnn.py
----------------
RCD training with GNN student — per-residue graph-based conditioning.
Directly regresses onto FlowPacker's cached vector fields.

Usage:
    python train_rcd_gnn.py \
        --outdir rcd-runs-gnn \
        --cache-dir datasets/vf_cache2 \
        --duration 1000 \
        --batch 32 \
        --lr 1e-3 \
        --seed 42
"""

import argparse
import copy
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flowpacker'))

from gnn_student import GNNStudent, build_graph, K_NEIGHBORS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'X':'X','UNK':'X',
}
ONE_LETTER = 'ACDEFGHIKLMNPQRSTVWYX'
LETTER_TO_NUM = {c: i for i, c in enumerate(ONE_LETTER)}


def _get_bb_dihedral(n, ca, c):
    L = ca.shape[0]
    d = torch.zeros(L, 3)
    def dih(a, b, cc, dd):
        b1=b-a; b2=cc-b; b3=dd-cc
        n1=torch.cross(b1,b2,dim=-1); n2=torch.cross(b2,b3,dim=-1)
        m1=torch.cross(n1,b2/(b2.norm(dim=-1,keepdim=True)+1e-8),dim=-1)
        return torch.atan2((m1*n2).sum(-1),(n1*n2).sum(-1))
    if L > 1:
        d[1:,0]=dih(c[:-1],n[1:],ca[1:],c[1:])
        d[:-1,1]=dih(n[:-1],ca[:-1],c[:-1],n[1:])
        d[1:,2]=dih(ca[:-1],c[:-1],n[1:],ca[1:])
    return d


class GNNCachedVFDataset(Dataset):
    """Per-protein dataset that returns everything needed to build a graph."""

    def __init__(self, cache_dir, side_chain_path, sequence_path, backbone_path, **kwargs):
        self.cache_dir   = cache_dir
        self.cache_files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.pt')])
        self.N = len(self.cache_files)
        print(f'GNNCachedVFDataset: {self.N} proteins')

        sc = np.load(side_chain_path)
        self.chi_mask = torch.tensor(sc['mask'], dtype=torch.float32)  # (N,512,4)

        seq = np.load(sequence_path, allow_pickle=True)
        self.names = seq['names']

        bb = np.load(backbone_path)
        self.backbone = torch.tensor(bb['backbone'], dtype=torch.float32)  # (N,512,4,3)
        self.bb_mask  = torch.tensor(bb['mask'],     dtype=torch.float32)  # (N,512)

    def __len__(self):
        return self.N

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
        vf_k = vf_all[k]        # (512,4)
        t_k  = t_all[k,:,0]    # (512,)
        xt_k = xt_all[k]        # (512,4)

        chi_mask = self.chi_mask[idx]   # (512,4)
        bb       = self.backbone[idx]   # (512,4,3)
        bb_mask  = self.bb_mask[idx]    # (512,)
        names    = self.names[idx]

        aa_num = torch.tensor(
            [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in names],
            dtype=torch.long)
        aa_onehot   = F.one_hot(aa_num, num_classes=21).float()
        bb_dihedral = _get_bb_dihedral(bb[:,0], bb[:,1], bb[:,2])

        return {
            'bb':          bb,           # (512,4,3)
            'bb_mask':     bb_mask,      # (512,)
            'aa_onehot':   aa_onehot,    # (512,21)
            'bb_dihedral': bb_dihedral,  # (512,3)
            'xt':          xt_k,         # (512,4)
            'vf':          vf_k,         # (512,4)
            't':           t_k,          # (512,)
            'chi_mask':    chi_mask,     # (512,4)
        }


# ---------------------------------------------------------------------------
# Training loop — process one protein at a time (graph can't be batched easily)
# ---------------------------------------------------------------------------

def fmt_time(s):
    s = int(s)
    if s < 60: return f'{s}s'
    elif s < 3600: return f'{s//60}m {s%60:02d}s'
    else: return f'{s//3600}h {(s%3600)//60:02d}m'


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = GNNCachedVFDataset(
        cache_dir=args.cache_dir,
        side_chain_path=args.side_chain_path,
        sequence_path=args.sequence_path,
        backbone_path=args.backbone_path,
    )
    # batch_size=1 — each protein is its own graph
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=False)

    net = GNNStudent().to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    n_params = sum(p.numel() for p in net.parameters())
    print(f'GNN student: {n_params/1e6:.2f}M parameters')

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.duration * 1000, eta_min=1e-5
    )

    total_epochs = args.duration
    os.makedirs(args.outdir, exist_ok=True)
    run_dir = os.path.join(args.outdir, f'rcd-gnn')
    os.makedirs(run_dir, exist_ok=True)
    print(f'Output: {run_dir}')

    log_file = open(os.path.join(run_dir, 'log.txt'), 'w')
    def log(msg):
        print(msg); log_file.write(msg+'\n'); log_file.flush()

    log(f'Training GNN student for {total_epochs} epochs...')

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(total_epochs):
        net.train()
        epoch_losses = []

        for batch in loader:
            bb       = batch['bb'].squeeze(0).to(device)          # (512,4,3)
            bb_mask  = batch['bb_mask'].squeeze(0).to(device)     # (512,)
            aa_oh    = batch['aa_onehot'].squeeze(0).to(device)   # (512,21)
            bb_dih   = batch['bb_dihedral'].squeeze(0).to(device) # (512,3)
            xt       = batch['xt'].squeeze(0).to(device)          # (512,4)
            vf_gt    = batch['vf'].squeeze(0).to(device)          # (512,4)
            t        = batch['t'].squeeze(0).to(device)           # (512,)
            chi_mask = batch['chi_mask'].squeeze(0).to(device)    # (512,4)

            # Build graph for valid residues
            node_feat, edge_index, edge_attr, valid = build_graph(
                bb, bb_mask, xt, t, aa_oh, bb_dih, k=K_NEIGHBORS, device=device
            )

            if node_feat is None or node_feat.shape[0] < 2:
                continue

            optimizer.zero_grad(set_to_none=True)

            pred_vf = net(node_feat, edge_index, edge_attr)  # (N_valid, 4)

            # Ground truth vf for valid residues
            vf_valid   = vf_gt[valid]       # (N_valid, 4)
            mask_valid = chi_mask[valid]    # (N_valid, 4)

            loss = ((pred_vf - vf_valid)**2 * mask_valid).sum(dim=-1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            elapsed  = time.time() - start_time
            lr_now   = optimizer.param_groups[0]['lr']
            msg = (f"epoch {epoch+1:<5d} loss {avg_loss:<9.5f} "
                   f"lr {lr_now:.2e} time {fmt_time(elapsed)}")
            log(msg)

            # Save checkpoint every 50 epochs and at end
            if (epoch+1) % 50 == 0 or epoch+1 == total_epochs:
                snap_path = os.path.join(run_dir, f'network-snapshot-epoch{epoch+1:04d}.pkl')
                data = {'ema': copy.deepcopy(ema).cpu(),
                        'net': copy.deepcopy(net).cpu(),
                        'epoch': epoch+1}
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)
                log(f'  Saved {snap_path}')

            # Update EMA
            ema_beta = 0.999
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    log('Training complete.')
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir',          default='rcd-runs-gnn')
    parser.add_argument('--cache-dir',       default='datasets/vf_cache2')
    parser.add_argument('--side-chain-path', default='datasets/side_chain_data.npz')
    parser.add_argument('--sequence-path',   default='datasets/sequence_data.npz')
    parser.add_argument('--backbone-path',   default='datasets/backbone_data.npz')
    parser.add_argument('--duration',        type=int,   default=200,
                        help='Number of epochs (not kimg)')
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--seed',            type=int,   default=42)
    args = parser.parse_args()
    train(args)