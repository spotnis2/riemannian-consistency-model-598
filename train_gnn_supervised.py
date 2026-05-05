"""
train_gnn_supervised.py
-----------------------
Supervised GNN training: predict chi angles directly from backbone structure.
No flow matching, no distillation.

Usage:
    python train_gnn_supervised.py \
        --outdir runs-gnn-supervised \
        --epochs 300 \
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

from gnn_chi_predictor import GNNChiPredictor, build_graph, K_NEIGHBORS

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
    d = torch.zeros(L, 3, device=ca.device)
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


class ProteinChiDataset(Dataset):
    """Loads backbone + sequence + chi angles for supervised training."""

    def __init__(self, backbone_path, side_chain_path, sequence_path,
                 train=True, test_split=0.2, seed=42, **kwargs):
        bb  = np.load(backbone_path)
        sc  = np.load(side_chain_path)
        seq = np.load(sequence_path, allow_pickle=True)

        self.backbone = torch.tensor(bb['backbone'], dtype=torch.float32)  # (N,512,4,3)
        self.bb_mask  = torch.tensor(bb['mask'],     dtype=torch.float32)  # (N,512)
        self.chi      = torch.tensor(sc['side_chains'], dtype=torch.float32)  # (N,512,4) [-pi,pi]
        self.chi_mask = torch.tensor(sc['mask'],        dtype=torch.float32)  # (N,512,4)
        self.names    = seq['names']  # (N,512)

        N = self.backbone.shape[0]
        torch.manual_seed(seed)
        idx = torch.randperm(N)
        n_test = int(N * test_split)

        if train:
            self.indices = idx[n_test:].tolist()
        else:
            self.indices = idx[:n_test].tolist()

        print(f"{'Train' if train else 'Test'} set: {len(self.indices)} proteins")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return {
            'backbone':    self.backbone[idx],   # (512,4,3)
            'bb_mask':     self.bb_mask[idx],    # (512,)
            'chi':         self.chi[idx],         # (512,4) in [-pi,pi]
            'chi_mask':    self.chi_mask[idx],   # (512,4)
            'names':       list(self.names[idx]),  # (512,) str
        }


def angular_loss(pred, target, mask):
    """
    Angular MSE loss on torus.
    pred, target in [0, 2pi]. mask: (N, 4).
    """
    diff = pred - target
    # wrap to [-pi, pi]
    diff = (diff + math.pi) % (2*math.pi) - math.pi
    loss = (diff**2 * mask).sum() / mask.sum().clamp(min=1)
    return loss


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

    # train_ds = ProteinChiDataset(args.backbone_path, args.side_chain_path,
    #                               args.sequence_path, train=True, seed=args.seed)
    # test_ds  = ProteinChiDataset(args.backbone_path, args.side_chain_path,
    #                               args.sequence_path, train=False, seed=args.seed)

    train_ds = ProteinChiDataset(args.backbone_path, args.side_chain_path,
                                  args.sequence_path, train=True, test_split=0.1, seed=args.seed)
    test_ds  = ProteinChiDataset(args.backbone_path, args.side_chain_path,
                                  args.sequence_path, train=False, test_split=0.1, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=False)

    net = GNNChiPredictor().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'GNN: {n_params/1e6:.2f}M parameters')

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    os.makedirs(args.outdir, exist_ok=True)
    log_file = open(os.path.join(args.outdir, 'log.txt'), 'w')
    def log(msg):
        print(msg); log_file.write(msg+'\n'); log_file.flush()

    log(f'Training GNN chi predictor for {args.epochs} epochs...')
    start_time = time.time()
    best_test_mae = float('inf')

    for epoch in range(args.epochs):
        # --- Train ---
        net.train()
        train_losses = []

        for batch in train_loader:
            bb       = batch['backbone'].squeeze(0).to(device)  # (512,4,3)
            bb_mask  = batch['bb_mask'].squeeze(0).to(device)   # (512,)
            chi_gt   = batch['chi'].squeeze(0).to(device)       # (512,4) [-pi,pi]
            chi_mask = batch['chi_mask'].squeeze(0).to(device)  # (512,4)
            names = [n[0] if isinstance(n, (list, tuple)) else str(n) for n in batch["names"]]                         # list of 512 strs

            # Build node features
            names = [n[0] if isinstance(n, (list, tuple)) else str(n) for n in batch["names"]]
            aa_num = torch.tensor(
                [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in names],
                dtype=torch.long, device=device)
            aa_oh  = F.one_hot(aa_num, num_classes=21).float().squeeze(0)
            bb_dih = _get_bb_dihedral(bb[:,0], bb[:,1], bb[:,2])

            node_feat, edge_index, edge_attr, valid = build_graph(
                bb, bb_mask, aa_oh, bb_dih, k=K_NEIGHBORS, device=device
            )
            if node_feat is None:
                continue

            optimizer.zero_grad(set_to_none=True)

            pred = net(node_feat, edge_index, edge_attr)  # (N_valid, 4) in [0,2pi]

            # Ground truth for valid residues, converted to [0,2pi]
            gt_valid   = (chi_gt[valid] % (2*math.pi))    # (N_valid, 4)
            mask_valid = chi_mask[valid]                   # (N_valid, 4)

            loss = angular_loss(pred, gt_valid, mask_valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # --- Evaluate every 10 epochs ---
        if (epoch+1) % 10 == 0 or epoch+1 == args.epochs:
            net.eval()
            all_diff, all_mask = [], []

            with torch.no_grad():
                for batch in test_loader:
                    bb       = batch['backbone'].squeeze(0).to(device)
                    bb_mask  = batch['bb_mask'].squeeze(0).to(device)
                    chi_gt   = batch['chi'].squeeze(0).to(device)
                    chi_mask = batch['chi_mask'].squeeze(0).to(device)
                    names = [n[0] if isinstance(n, (list, tuple)) else str(n) for n in batch["names"]]

                    aa_num = torch.tensor(
                        [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in names],
                        dtype=torch.long, device=device)
                    aa_oh  = F.one_hot(aa_num, num_classes=21).float().squeeze(0)
                    bb_dih = _get_bb_dihedral(bb[:,0], bb[:,1], bb[:,2])

                    node_feat, edge_index, edge_attr, valid = build_graph(
                        bb, bb_mask, aa_oh, bb_dih, k=K_NEIGHBORS, device=device
                    )
                    if node_feat is None:
                        continue

                    pred = net(node_feat, edge_index, edge_attr)  # (N_valid, 4)
                    gt_valid   = (chi_gt[valid] % (2*math.pi))
                    mask_valid = chi_mask[valid]

                    diff = pred - gt_valid
                    diff = (diff + math.pi) % (2*math.pi) - math.pi
                    diff_deg = diff.abs() * (180.0/math.pi)

                    all_diff.append(diff_deg.cpu())
                    all_mask.append(mask_valid.cpu().bool())

            all_diff = torch.cat(all_diff, dim=0)
            all_mask = torch.cat(all_mask, dim=0)
            valid_diff = all_diff[all_mask]
            test_mae = valid_diff.mean().item()
            test_acc = (valid_diff < 20.0).float().mean().item() * 100

            avg_train = np.mean(train_losses) if train_losses else 0
            lr_now = optimizer.param_groups[0]['lr']
            log(f"epoch {epoch+1:<5d} train_loss {avg_train:<8.5f} "
                f"test_mae {test_mae:<7.2f}° test_acc {test_acc:<6.2f}% "
                f"lr {lr_now:.2e} time {fmt_time(time.time()-start_time)}")

            # Save best
            if test_mae < best_test_mae:
                best_test_mae = test_mae
                snap_path = os.path.join(args.outdir, 'best-model.pkl')
                data = {'net': copy.deepcopy(net).cpu(), 'epoch': epoch+1,
                        'test_mae': test_mae, 'test_acc': test_acc}
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)
                log(f'  ** New best: {test_mae:.2f}° — saved to {snap_path}')

            # Save periodic checkpoint
            if (epoch+1) % 50 == 0 or epoch+1 == args.epochs:
                snap_path = os.path.join(args.outdir, f'checkpoint-epoch{epoch+1:04d}.pkl')
                data = {'net': copy.deepcopy(net).cpu(), 'epoch': epoch+1}
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)

        elif train_losses:
            avg_train = np.mean(train_losses)
            log(f"epoch {epoch+1:<5d} train_loss {avg_train:<8.5f} "
                f"lr {optimizer.param_groups[0]['lr']:.2e} "
                f"time {fmt_time(time.time()-start_time)}")

    log(f'\nTraining complete. Best test MAE: {best_test_mae:.2f}°')
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir',          default='runs-gnn-supervised')
    parser.add_argument('--backbone-path',   default='datasets/backbone_data.npz')
    parser.add_argument('--side-chain-path', default='datasets/side_chain_data.npz')
    parser.add_argument('--sequence-path',   default='datasets/sequence_data.npz')
    parser.add_argument('--epochs',          type=int,   default=300)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--seed',            type=int,   default=42)
    args = parser.parse_args()
    train(args)