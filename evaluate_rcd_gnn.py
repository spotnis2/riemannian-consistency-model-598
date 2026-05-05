"""
evaluate_rcd_gnn.py — Evaluation for GNN student RCD model.
Samples by Euler integration using the GNN's predicted vf.

Usage:
    python evaluate_rcd_gnn.py \
        --checkpoint rcd-runs-gnn/rcd-gnn/network-snapshot-epoch0200.pkl
"""

import argparse
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flowpacker'))

from gnn_student import GNNStudent, build_graph, K_NEIGHBORS
from train_rcd_gnn import _get_bb_dihedral, THREE_TO_ONE, LETTER_TO_NUM


def angular_diff_deg(pred, target):
    diff = pred - target
    diff = (diff + math.pi) % (2*math.pi) - math.pi
    return torch.abs(diff) * (180.0/math.pi)


def build_static_graph(bb, bb_mask, aa_oh, bb_dih, device):
    """Build graph without xt/t — used to get structure, then we inject xt/t per step."""
    ca = bb[bb_mask.bool(), 1, :]
    return ca, bb_mask.bool()


@torch.no_grad()
def sample_euler_gnn(net, bb, bb_mask, aa_oh, bb_dih, device, n_steps=20):
    """
    Euler integration for one protein.
    Returns predicted chi angles for valid residues.
    """
    valid = bb_mask.bool()
    N_valid = valid.sum().item()

    if N_valid == 0:
        return torch.zeros(0, 4)

    # Start from uniform noise
    xt_full = torch.rand(512, 4, device=device) * 2*math.pi

    ts = torch.linspace(1.0, 0.0, n_steps+1, device=device)
    dt = ts[0] - ts[1]

    for i in range(n_steps):
        t_val = ts[i]
        t_full = torch.full((512,), t_val.item(), device=device)

        node_feat, edge_index, edge_attr, _ = build_graph(
            bb, bb_mask, xt_full, t_full, aa_oh, bb_dih,
            k=K_NEIGHBORS, device=device
        )

        if node_feat is None:
            break

        vf_valid = net(node_feat, edge_index, edge_attr)  # (N_valid, 4)

        # Update only valid residues
        xt_full[valid] = (xt_full[valid] + dt * vf_valid) % (2*math.pi)

    return xt_full[valid].cpu()  # (N_valid, 4)


def evaluate(net, sc_path, backbone_path, sequence_path, device,
             n_steps=20, test_split=0.2, seed=42):
    sc     = np.load(sc_path)
    angles = torch.tensor(sc['side_chains'], dtype=torch.float32)  # (N,512,4) [-pi,pi]
    chi_valid = (angles.abs() > 1e-6).float()
    res_valid = chi_valid.any(dim=-1).float()

    bb_data = np.load(backbone_path)
    backbone = torch.tensor(bb_data['backbone'], dtype=torch.float32)  # (N,512,4,3)
    bb_mask  = torch.tensor(bb_data['mask'],     dtype=torch.float32)  # (N,512)

    seq_data = np.load(sequence_path, allow_pickle=True)
    names    = seq_data['names']

    N = angles.shape[0]
    n_test = max(1, int(N*test_split))
    torch.manual_seed(seed)
    test_idx = torch.randperm(N)[:n_test].tolist()

    all_pred, all_gt, all_chimask = [], [], []

    net.eval()
    for i, idx in enumerate(test_idx):
        if i % 20 == 0:
            print(f"  Evaluating protein {i}/{n_test}...")

        bb_i   = backbone[idx].to(device)
        mask_i = bb_mask[idx].to(device)
        nm_i   = names[idx]

        aa_num = torch.tensor(
            [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in nm_i],
            dtype=torch.long)
        aa_oh  = F.one_hot(aa_num, num_classes=21).float().to(device)
        bb_dih = _get_bb_dihedral(bb_i[:,0], bb_i[:,1], bb_i[:,2]).to(device)

        pred = sample_euler_gnn(net, bb_i, mask_i, aa_oh, bb_dih, device, n_steps=n_steps)

        valid    = mask_i.bool().cpu()
        gt_valid = angles[idx][valid]           # (N_valid, 4) [-pi,pi]
        cm_valid = chi_valid[idx][valid]        # (N_valid, 4)

        all_pred.append(pred)
        all_gt.append(gt_valid)
        all_chimask.append(cm_valid)

    pred_flat = torch.cat(all_pred, dim=0)
    gt_flat   = torch.cat(all_gt,   dim=0)
    cm_flat   = torch.cat(all_chimask, dim=0).bool()

    gt_wrapped = gt_flat % (2*math.pi)
    diff_deg   = angular_diff_deg(pred_flat, gt_wrapped)

    all_diff = diff_deg[cm_flat]
    mae_all  = all_diff.mean().item()
    acc_all  = (all_diff < 20.0).float().mean().item() * 100

    print(f"\n{'='*55}")
    print(f"Results ({n_steps}-step Euler, GNN student)")
    print(f"{'='*55}")
    print(f"Overall  MAE: {mae_all:.2f}°   Accuracy: {acc_all:.2f}%")
    print(f"\nPer-chi:")
    print(f"  {'Chi':<8} {'MAE (°)':<12} {'Acc (%)':<12} N")
    print(f"  {'-'*45}")
    for i, name in enumerate(['chi1','chi2','chi3','chi4']):
        mi = cm_flat[:,i]
        if mi.sum() == 0:
            print(f"  {name:<8} {'N/A':<12} {'N/A':<12} 0")
            continue
        d = diff_deg[:,i][mi]
        print(f"  {name:<8} {d.mean().item():<12.2f} "
              f"{(d<20).float().mean().item()*100:<12.2f} {mi.sum().item()}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--sc-path',       default='datasets/side_chain_data.npz')
    parser.add_argument('--backbone-path', default='datasets/backbone_data.npz')
    parser.add_argument('--sequence-path', default='datasets/sequence_data.npz')
    parser.add_argument('--n-steps',       type=int,   default=20)
    parser.add_argument('--test-split',    type=float, default=0.2)
    parser.add_argument('--seed',          type=int,   default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        data = pickle.load(f)
    net = data['net'].to(device).eval()
    print(f"Loaded.")

    evaluate(net, args.sc_path, args.backbone_path, args.sequence_path,
             device, n_steps=args.n_steps, test_split=args.test_split,
             seed=args.seed)


if __name__ == '__main__':
    main()