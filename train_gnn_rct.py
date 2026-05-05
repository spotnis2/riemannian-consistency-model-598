"""
train_gnn_rct.py
----------------
Riemannian Consistency Training (RCT) with GNN student.
Warm-started from pretrained supervised GNN checkpoint.

RCT uses a student/teacher setup where:
  - Student: GNN at current weights
  - Teacher: EMA of the student (no external teacher needed)
  - Loss: consistency loss — student at t should agree with teacher at t+dt

This is pure RCT (no FlowPacker), warm-started from supervised GNN.
Goal: achieve 1-step generation matching supervised GNN's multi-step accuracy.

Usage:
    python train_gnn_rct.py \
        --pretrained runs-gnn-supervised/best-model.pkl \
        --outdir runs-gnn-rct \
        --epochs 200 \
        --lr 5e-5 \
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
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flowpacker'))

from gnn_chi_predictor import GNNChiPredictor, build_graph, K_NEIGHBORS, NODE_IN_DIM, EDGE_IN_DIM, HIDDEN_DIM, N_LAYERS
from train_gnn_supervised import ProteinChiDataset, _get_bb_dihedral, THREE_TO_ONE, LETTER_TO_NUM


def angular_diff(pred, target):
    """Shortest angular distance on torus, in radians."""
    diff = pred - target
    return (diff + math.pi) % (2*math.pi) - math.pi


def rct_loss(student, teacher_ema, node_feat_base, edge_index, edge_attr,
             chi_gt, chi_mask, valid, n_steps=18, eps=0.05):
    """
    Riemannian Consistency Training loss.

    Sample two adjacent timesteps t1 < t2, generate xt1 and xt2
    from the same noise, then enforce:
        student(xt1, t1) ≈ teacher_ema(xt2, t2).detach()

    Args:
        student:        current GNN
        teacher_ema:    EMA copy of student
        node_feat_base: (N, 27) backbone features
        edge_index:     (2, E)
        edge_attr:      (E, 7)
        chi_gt:         (512, 4) ground truth chi angles [-pi, pi]
        chi_mask:       (512, 4) per-chi validity mask
        valid:          (512,) bool — valid residues
        n_steps:        number of discretization steps
        eps:            minimum timestep
    """
    N = node_feat_base.shape[0]
    device = node_feat_base.device

    # Sample adjacent timestep indices
    # t schedule: eps, eps + 1/n, ..., 1
    step = torch.randint(0, n_steps - 1, (1,), device=device).item()
    t1 = eps + step     / n_steps * (1 - eps)
    t2 = eps + (step+1) / n_steps * (1 - eps)

    # Ground truth for valid residues in [0, 2pi]
    x1       = chi_gt[valid] % (2*math.pi)   # (N, 4)
    mask     = chi_mask[valid]                # (N, 4)

    # Sample noise
    x0 = torch.rand(N, 4, device=device) * 2*math.pi

    # Interpolate on torus: xt = (x0 + t * log_x0_x1) % 2pi
    log_x0_x1 = torch.atan2(torch.sin(x1 - x0), torch.cos(x1 - x0))
    xt1 = (x0 + t1 * log_x0_x1) % (2*math.pi)
    xt2 = (x0 + t2 * log_x0_x1) % (2*math.pi)

    # Build augmented node features with xt and t
    t1_feat = torch.full((N, 1), t1, device=device)
    t2_feat = torch.full((N, 1), t2, device=device)
    node_t1 = torch.cat([node_feat_base, xt1, t1_feat], dim=-1)  # (N, 32)
    node_t2 = torch.cat([node_feat_base, xt2, t2_feat], dim=-1)  # (N, 32)

    # Student predicts x1 from xt1 at t1
    x1_student = student(node_t1, edge_index, edge_attr)  # (N, 4)

    # Teacher (EMA) predicts x1 from xt2 at t2
    with torch.no_grad():
        x1_teacher = teacher_ema(node_t2, edge_index, edge_attr)  # (N, 4)

    # Consistency loss: student and teacher should agree
    diff = angular_diff(x1_student, x1_teacher.detach())  # (N, 4)

    # Weight by 1/(t2-t1) to normalize across timestep gaps
    dt = t2 - t1
    loss = (diff**2 * mask).sum() / mask.sum().clamp(min=1) / dt

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

    # Load pretrained supervised GNN and build RCT student (32-dim input)
    print(f'Loading pretrained model from {args.pretrained}...')
    with open(args.pretrained, 'rb') as f:
        ckpt = pickle.load(f)

    # RCT student needs 27+4+1=32 dim input (backbone + xt + t)
    student = GNNChiPredictor(
        node_in=NODE_IN_DIM + 5,  # 32
        edge_in=EDGE_IN_DIM,
        hidden=HIDDEN_DIM,
        n_layers=N_LAYERS,
    ).to(device)

    pretrained_net = ckpt['net']
    # Transfer pretrained weights, padding node_embed for new input dims
    student_dict    = student.state_dict()
    pretrained_dict = pretrained_net.state_dict()
    transferred = 0
    for k, v in pretrained_dict.items():
        if k in student_dict and student_dict[k].shape == v.shape:
            student_dict[k] = v
            transferred += 1
        elif k == 'node_embed.weight':
            # Pad from (hidden, 27) to (hidden, 32) with zeros for new xt+t dims
            padded = torch.zeros_like(student_dict[k])
            padded[:, :v.shape[1]] = v  # copy pretrained 27 dims
            student_dict[k] = padded
            transferred += 1
            print(f'Padded node_embed.weight: {v.shape} -> {padded.shape}')
    student.load_state_dict(student_dict)
    print(f'Transferred {transferred} weight tensors (node_embed padded, not reset).')

    # EMA teacher starts as exact copy of student
    teacher_ema = copy.deepcopy(student).eval().requires_grad_(False)

    n_params = sum(p.numel() for p in student.parameters())
    print(f'RCT GNN: {n_params/1e6:.2f}M parameters')

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    train_ds = ProteinChiDataset(args.backbone_path, args.side_chain_path,
                                  args.sequence_path, train=True,  seed=args.seed)
    test_ds  = ProteinChiDataset(args.backbone_path, args.side_chain_path,
                                  args.sequence_path, train=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=False)

    os.makedirs(args.outdir, exist_ok=True)
    log_file = open(os.path.join(args.outdir, 'log.txt'), 'w')
    def log(msg):
        print(msg); log_file.write(msg+'\n'); log_file.flush()

    log(f'RCT fine-tuning for {args.epochs} epochs, n_steps={args.n_steps}, lr={args.lr}...')
    start_time = time.time()
    best_mae = float('inf')

    for epoch in range(args.epochs):
        student.train()
        train_losses = []

        for batch in train_loader:
            bb       = batch['backbone'].squeeze(0).to(device)
            bb_mask  = batch['bb_mask'].squeeze(0).to(device)
            chi_gt   = batch['chi'].squeeze(0).to(device)
            chi_mask = batch['chi_mask'].squeeze(0).to(device)
            names    = [n[0] if isinstance(n,(list,tuple)) else str(n) for n in batch['names']]

            aa_num = torch.tensor(
                [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in names],
                dtype=torch.long, device=device)
            aa_oh  = F.one_hot(aa_num, num_classes=21).float()
            bb_dih = _get_bb_dihedral(bb[:,0], bb[:,1], bb[:,2])

            node_feat_base, edge_index, edge_attr, valid = build_graph(
                bb, bb_mask, aa_oh, bb_dih, k=K_NEIGHBORS, device=device
            )
            if node_feat_base is None or node_feat_base.shape[0] < 4:
                continue

            optimizer.zero_grad(set_to_none=True)

            loss = rct_loss(
                student, teacher_ema,
                node_feat_base, edge_index, edge_attr,
                chi_gt, chi_mask, valid,
                n_steps=args.n_steps,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

            # EMA update — slower decay for stability
            ema_beta = 0.9999
            for p_ema, p_stu in zip(teacher_ema.parameters(), student.parameters()):
                p_ema.copy_(p_stu.detach().lerp(p_ema, ema_beta))

        scheduler.step()

        # Evaluate every 10 epochs — 1-step sampling
        if (epoch+1) % 10 == 0 or epoch+1 == args.epochs:
            student.eval()
            all_diff, all_mask = [], []

            with torch.no_grad():
                for batch in test_loader:
                    bb       = batch['backbone'].squeeze(0).to(device)
                    bb_mask  = batch['bb_mask'].squeeze(0).to(device)
                    chi_gt   = batch['chi'].squeeze(0).to(device)
                    chi_mask = batch['chi_mask'].squeeze(0).to(device)
                    names    = [n[0] if isinstance(n,(list,tuple)) else str(n) for n in batch['names']]

                    aa_num = torch.tensor(
                        [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in names],
                        dtype=torch.long, device=device)
                    aa_oh  = F.one_hot(aa_num, num_classes=21).float()
                    bb_dih = _get_bb_dihedral(bb[:,0], bb[:,1], bb[:,2])

                    node_feat_base, edge_index, edge_attr, valid = build_graph(
                        bb, bb_mask, aa_oh, bb_dih, k=K_NEIGHBORS, device=device
                    )
                    if node_feat_base is None:
                        continue

                    N = node_feat_base.shape[0]

                    # 1-step: pure noise at t=1, predict x1
                    x0     = torch.rand(N, 4, device=device) * 2*math.pi
                    t_feat = torch.ones(N, 1, device=device)
                    node_t = torch.cat([node_feat_base, x0, t_feat], dim=-1)
                    pred   = student(node_t, edge_index, edge_attr)  # (N, 4)

                    gt_valid   = (chi_gt[valid] % (2*math.pi))
                    mask_valid = chi_mask[valid]

                    diff     = angular_diff(pred, gt_valid)
                    diff_deg = diff.abs() * (180.0/math.pi)
                    all_diff.append(diff_deg.cpu())
                    all_mask.append(mask_valid.cpu().bool())

            all_diff = torch.cat(all_diff, dim=0)
            all_mask = torch.cat(all_mask, dim=0)
            valid_diff = all_diff[all_mask]
            test_mae = valid_diff.mean().item()
            test_acc = (valid_diff < 20.0).float().mean().item() * 100

            avg_train = np.mean(train_losses) if train_losses else 0
            log(f"epoch {epoch+1:<5d} train_loss {avg_train:<8.5f} "
                f"test_mae {test_mae:<7.2f}° test_acc {test_acc:<6.2f}% "
                f"[1-step] time {fmt_time(time.time()-start_time)}")

            if test_mae < best_mae:
                best_mae = test_mae
                snap_path = os.path.join(args.outdir, 'best-model.pkl')
                data = {'net': copy.deepcopy(student).cpu(),
                        'ema': copy.deepcopy(teacher_ema).cpu(),
                        'epoch': epoch+1,
                        'test_mae': test_mae,
                        'test_acc': test_acc}
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)
                log(f'  ** New best: {test_mae:.2f}° / {test_acc:.2f}% — saved')

            if (epoch+1) % 50 == 0 or epoch+1 == args.epochs:
                snap_path = os.path.join(args.outdir, f'checkpoint-epoch{epoch+1:04d}.pkl')
                data = {'net': copy.deepcopy(student).cpu(), 'epoch': epoch+1}
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)
        else:
            avg_train = np.mean(train_losses) if train_losses else 0
            log(f"epoch {epoch+1:<5d} train_loss {avg_train:<8.5f} "
                f"time {fmt_time(time.time()-start_time)}")

    log(f'\nRCT fine-tuning complete. Best 1-step MAE: {best_mae:.2f}°')
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained',      required=True)
    parser.add_argument('--outdir',          default='runs-gnn-rct')
    parser.add_argument('--backbone-path',   default='datasets/backbone_data.npz')
    parser.add_argument('--side-chain-path', default='datasets/side_chain_data.npz')
    parser.add_argument('--sequence-path',   default='datasets/sequence_data.npz')
    parser.add_argument('--epochs',          type=int,   default=200)
    parser.add_argument('--n-steps',         type=int,   default=18,
                        help='Number of consistency discretization steps')
    parser.add_argument('--lr',              type=float, default=5e-5)
    parser.add_argument('--seed',            type=int,   default=42)
    args = parser.parse_args()
    train(args)