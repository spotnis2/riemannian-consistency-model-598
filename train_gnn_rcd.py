"""
train_gnn_rcd.py
----------------
Riemannian Consistency Distillation fine-tuning for the pretrained GNN.

The GNN (pretrained supervised) is fine-tuned to be consistent across timesteps:
  - Teacher: EMA copy of the GNN (self-distillation, no FlowPacker needed)
  - Student: the GNN itself
  - Loss: consistency loss — predictions at nearby timesteps should agree

After fine-tuning, the GNN can generate chi angles in a SINGLE forward pass
instead of 20 Euler steps.

Usage:
    python train_gnn_rcd.py \
        --pretrained runs-gnn-supervised/best-model.pkl \
        --outdir runs-gnn-rcd \
        --epochs 100 \
        --lr 1e-4 \
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

from gnn_chi_predictor import GNNChiPredictor, build_graph, K_NEIGHBORS
from train_gnn_supervised import ProteinChiDataset, _get_bb_dihedral, THREE_TO_ONE, LETTER_TO_NUM


def angular_diff(pred, target):
    """Shortest angular distance on torus. Returns (N, 4) in radians."""
    diff = pred - target
    return (diff + math.pi) % (2*math.pi) - math.pi


def consistency_loss_gnn(student, teacher_ema, node_feat, edge_index, edge_attr,
                          chi_gt, chi_mask, valid, dt=0.1, eps=0.05):
    """
    Consistency loss for GNN.

    At timestep t, student predicts x1.
    At timestep t+dt, EMA teacher also predicts x1 from a slightly noisier xt.
    They should agree.

    Steps:
    1. Sample t ~ Uniform[eps, 1-dt]
    2. Generate xt = interpolate(noise, x1, t)
    3. Student predicts x1_s from xt at t
    4. Generate xt_dt = interpolate(noise, x1, t+dt)  
    5. Teacher predicts x1_t from xt_dt at t+dt
    6. Loss: |x1_s - x1_t.detach()|^2 on torus
    """
    N = node_feat.shape[0]
    device = node_feat.device

    # Sample timestep
    t = torch.rand(1, device=device) * (1 - dt - eps) + eps
    t_next = t + dt

    # Ground truth chi for valid residues, in [0, 2pi]
    x1 = chi_gt[valid] % (2*math.pi)          # (N, 4)
    mask = chi_mask[valid]                      # (N, 4)

    # Sample noise
    x0 = torch.rand(N, 4, device=device) * 2*math.pi

    # Interpolate on torus: xt = x0 + t * log(x0, x1)  mod 2pi
    log_x0_x1 = torch.atan2(torch.sin(x1 - x0), torch.cos(x1 - x0))
    xt      = (x0 + t      * log_x0_x1) % (2*math.pi)
    xt_next = (x0 + t_next * log_x0_x1) % (2*math.pi)

    # Build node features with xt injected
    # Replace xt portion of node features (last 4 dims... wait, supervised GNN doesn't take xt)
    # We need to add xt and t to node features for consistency model
    t_feat      = torch.full((N, 1), t.item(),      device=device)
    t_next_feat = torch.full((N, 1), t_next.item(), device=device)

    # Augment node features with xt and t
    node_feat_t      = torch.cat([node_feat, xt,      t_feat],      dim=-1)  # (N, 27+4+1=32)
    node_feat_t_next = torch.cat([node_feat, xt_next, t_next_feat], dim=-1)  # (N, 32)

    # Student prediction at t
    x1_student = student(node_feat_t, edge_index, edge_attr)       # (N, 4)

    # Teacher (EMA) prediction at t+dt
    with torch.no_grad():
        x1_teacher = teacher_ema(node_feat_t_next, edge_index, edge_attr)  # (N, 4)

    # Consistency loss: student and teacher should agree on x1
    diff = angular_diff(x1_student, x1_teacher.detach())  # (N, 4)
    loss = (diff**2 * mask).sum() / mask.sum().clamp(min=1)

    return loss, x1_student


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

    # Load pretrained GNN
    print(f'Loading pretrained model from {args.pretrained}...')
    with open(args.pretrained, 'rb') as f:
        ckpt = pickle.load(f)
    
    # The supervised GNN has 27-dim node input, but RCD needs 27+4+1=32
    # So we build a new GNN with 32-dim input and copy weights where possible
    from gnn_chi_predictor import NODE_IN_DIM, EDGE_IN_DIM, HIDDEN_DIM, N_LAYERS
    
    student = GNNChiPredictor(
        node_in=NODE_IN_DIM + 5,  # 27 + xt(4) + t(1) = 32
        edge_in=EDGE_IN_DIM,
        hidden=HIDDEN_DIM,
        n_layers=N_LAYERS,
    ).to(device)

    # Copy pretrained weights except node_embed (different input dim)
    pretrained_net = ckpt['net']
    student_dict   = student.state_dict()
    pretrained_dict = pretrained_net.state_dict()
    
    # Copy all layers except node_embed.weight/bias
    for k, v in pretrained_dict.items():
        if k in student_dict and student_dict[k].shape == v.shape:
            student_dict[k] = v
    student.load_state_dict(student_dict)
    print(f'Loaded pretrained weights (node_embed re-initialized for new input dim)')

    # EMA teacher — starts as copy of student
    teacher_ema = copy.deepcopy(student).eval().requires_grad_(False)

    n_params = sum(p.numel() for p in student.parameters())
    print(f'GNN student: {n_params/1e6:.2f}M parameters')

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    train_ds = ProteinChiDataset(args.backbone_path, args.side_chain_path,
                                  args.sequence_path, train=True, seed=args.seed)
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

    log(f'RCD fine-tuning for {args.epochs} epochs, dt={args.dt}...')
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
            names    = [n[0] if isinstance(n, (list,tuple)) else str(n) for n in batch['names']]

            aa_num = torch.tensor(
                [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in names],
                dtype=torch.long, device=device)
            aa_oh  = F.one_hot(aa_num, num_classes=21).float()
            bb_dih = _get_bb_dihedral(bb[:,0], bb[:,1], bb[:,2])

            # Build base graph (without xt/t — added inside loss fn)
            node_feat_base, edge_index, edge_attr, valid = build_graph(
                bb, bb_mask, aa_oh, bb_dih, k=K_NEIGHBORS, device=device
            )
            if node_feat_base is None or node_feat_base.shape[0] < 4:
                continue

            optimizer.zero_grad(set_to_none=True)

            loss, _ = consistency_loss_gnn(
                student, teacher_ema,
                node_feat_base, edge_index, edge_attr,
                chi_gt, chi_mask, valid, dt=args.dt
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

            # EMA update
            ema_beta = 0.999
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

                    # 1-step: start from noise at t=1, predict x1 directly
                    x0 = torch.rand(N, 4, device=device) * 2*math.pi
                    t_feat = torch.ones(N, 1, device=device)
                    node_feat_t = torch.cat([node_feat_base, x0, t_feat], dim=-1)
                    pred = student(node_feat_t, edge_index, edge_attr)  # (N, 4)

                    gt_valid   = (chi_gt[valid] % (2*math.pi))
                    mask_valid = chi_mask[valid]

                    diff = angular_diff(pred, gt_valid)
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
                        'epoch': epoch+1, 'test_mae': test_mae, 'test_acc': test_acc}
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)
                log(f'  ** New best: {test_mae:.2f}° — saved')

            if (epoch+1) % 50 == 0 or epoch+1 == args.epochs:
                snap_path = os.path.join(args.outdir, f'checkpoint-epoch{epoch+1:04d}.pkl')
                data = {'net': copy.deepcopy(student).cpu(), 'epoch': epoch+1}
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)
        else:
            avg_train = np.mean(train_losses) if train_losses else 0
            log(f"epoch {epoch+1:<5d} train_loss {avg_train:<8.5f} "
                f"time {fmt_time(time.time()-start_time)}")

    log(f'\nRCD fine-tuning complete. Best 1-step MAE: {best_mae:.2f}°')
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained',      required=True,
                        help='Path to pretrained supervised GNN checkpoint')
    parser.add_argument('--outdir',          default='runs-gnn-rcd')
    parser.add_argument('--backbone-path',   default='datasets/backbone_data.npz')
    parser.add_argument('--side-chain-path', default='datasets/side_chain_data.npz')
    parser.add_argument('--sequence-path',   default='datasets/sequence_data.npz')
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--dt',              type=float, default=0.05,
                        help='Consistency distillation timestep gap')
    parser.add_argument('--seed',            type=int,   default=42)
    args = parser.parse_args()
    train(args)