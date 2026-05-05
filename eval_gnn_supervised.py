"""
eval_gnn_supervised.py
----------------------
Evaluate the supervised GNN chi predictor using multi-step Euler integration.

Usage:
    python eval_gnn_supervised.py \
        --checkpoint runs-gnn-supervised/best-model.pkl \
        --n-steps 20
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

from gnn_chi_predictor import GNNChiPredictor, build_graph, K_NEIGHBORS
from train_gnn_supervised import _get_bb_dihedral, THREE_TO_ONE, LETTER_TO_NUM


def angular_diff_deg(pred, target):
    diff = pred - target
    diff = (diff + math.pi) % (2*math.pi) - math.pi
    return diff.abs() * (180.0/math.pi)


@torch.no_grad()
def sample_euler(net, node_feat_base, edge_index, edge_attr, device, n_steps=20):
    """
    Euler integration: start from noise at t=1, integrate to t=0.
    net: supervised GNN — predicts x1 from (node_feat + xt + t)
    """
    N = node_feat_base.shape[0]
    x = torch.rand(N, 4, device=device) * 2*math.pi  # start from noise

    ts = torch.linspace(1.0, 0.0, n_steps+1, device=device)
    dt = ts[0] - ts[1]

    for i in range(n_steps):
        t_val = ts[i]
        t_feat = torch.full((N, 1), t_val.item(), device=device)
        node_feat_t = node_feat_base  # supervised GNN: no xt/t input

        x1_pred = net(node_feat_t, edge_index, edge_attr)  # (N, 4) predicted clean

        # Move toward x1_pred
        # vf = log(xt, x1) / (1-t) = direction from xt toward x1
        log_x_x1 = torch.atan2(torch.sin(x1_pred - x), torch.cos(x1_pred - x))
        vf = log_x_x1 / (1 - t_val + 1e-8)

        x = (x + dt * vf) % (2*math.pi)

    return x  # (N, 4)


@torch.no_grad()
def sample_1step(net, node_feat_base, edge_index, edge_attr, device):
    """1-step: start from noise at t=1, predict x1 directly."""
    N = node_feat_base.shape[0]
    x0 = torch.rand(N, 4, device=device) * 2*math.pi
    t_feat = torch.ones(N, 1, device=device)
    node_feat_t = node_feat_base  # supervised GNN: no xt/t input
    return net(node_feat_t, edge_index, edge_attr)  # (N, 4)


def evaluate(net, backbone_path, side_chain_path, sequence_path,
             device, n_steps=20, test_split=0.2, seed=42):

    bb_data  = np.load(backbone_path)
    sc_data  = np.load(side_chain_path)
    seq_data = np.load(sequence_path, allow_pickle=True)

    backbone = torch.tensor(bb_data['backbone'], dtype=torch.float32)  # (N,512,4,3)
    bb_mask  = torch.tensor(bb_data['mask'],     dtype=torch.float32)  # (N,512)
    chi_gt   = torch.tensor(sc_data['side_chains'], dtype=torch.float32)  # (N,512,4)
    chi_mask = torch.tensor(sc_data['mask'],        dtype=torch.float32)  # (N,512,4)
    names    = seq_data['names']  # (N,512)

    # Per-chi validity from nonzero angles
    chi_valid = (chi_gt.abs() > 1e-6).float()
    res_valid = chi_valid.any(dim=-1).float()

    N = backbone.shape[0]
    n_test = max(1, int(N * test_split))
    torch.manual_seed(seed)
    test_idx = torch.randperm(N)[:n_test].tolist()

    net.eval()

    def run_eval(n_steps_eval, label):
        all_pred, all_gt, all_chimask = [], [], []

        for ii, idx in enumerate(test_idx):
            if ii % 50 == 0:
                print(f"  [{label}] {ii}/{n_test}...")

            bb_i   = backbone[idx].to(device)
            mask_i = bb_mask[idx].to(device)
            nm_i   = names[idx]

            aa_num = torch.tensor(
                [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in nm_i],
                dtype=torch.long, device=device)
            aa_oh  = F.one_hot(aa_num, num_classes=21).float()
            bb_dih = _get_bb_dihedral(bb_i[:,0], bb_i[:,1], bb_i[:,2])

            node_feat_base, edge_index, edge_attr, valid = build_graph(
                bb_i, mask_i, aa_oh, bb_dih, k=K_NEIGHBORS, device=device
            )
            if node_feat_base is None:
                continue

            if n_steps_eval == 1:
                pred = sample_1step(net, node_feat_base, edge_index, edge_attr, device)
            else:
                pred = sample_euler(net, node_feat_base, edge_index, edge_attr,
                                    device, n_steps=n_steps_eval)

            valid_cpu   = valid.cpu()
            gt_valid    = (chi_gt[idx][valid_cpu] % (2*math.pi))
            cm_valid    = chi_valid[idx][valid_cpu]

            all_pred.append(pred.cpu())
            all_gt.append(gt_valid)
            all_chimask.append(cm_valid)

        pred_flat = torch.cat(all_pred,    dim=0)
        gt_flat   = torch.cat(all_gt,      dim=0)
        cm_flat   = torch.cat(all_chimask, dim=0).bool()

        diff_deg  = angular_diff_deg(pred_flat, gt_flat)
        all_diff  = diff_deg[cm_flat]
        mae_all   = all_diff.mean().item()
        acc_all   = (all_diff < 20.0).float().mean().item() * 100

        print(f"\n{'='*55}")
        print(f"Results — {label}")
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
        return mae_all, acc_all

    # Run for requested n_steps and also 1-step for comparison
    run_eval(n_steps, f"{n_steps}-step Euler")
    if n_steps != 1:
        run_eval(1, "1-step (direct prediction)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--backbone-path', default='datasets/backbone_data.npz')
    parser.add_argument('--sc-path',       default='datasets/side_chain_data.npz')
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
    # Try 'net' first, then 'ema'
    net = data.get('net', data.get('ema')).to(device).eval()
    print(f"Loaded. Epoch: {data.get('epoch', '?')}")

    evaluate(net,
             backbone_path=args.backbone_path,
             side_chain_path=args.sc_path,
             sequence_path=args.sequence_path,
             device=device,
             n_steps=args.n_steps,
             test_split=args.test_split,
             seed=args.seed)


if __name__ == '__main__':
    main()