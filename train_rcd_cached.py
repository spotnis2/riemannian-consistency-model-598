"""
train_rcd_cached.py
-------------------
RCD training using precomputed FlowPacker vector fields.
Flow matching loss: student directly regresses onto teacher vf.
27-dim rotation-invariant per-residue conditioning.

Usage:
    python train_rcd_cached.py \
        --outdir rcd-runs-cached \
        --cache-dir datasets/vf_cache2 \
        --duration 200 --batch 512 --lr 8e-4 --seed 42
"""

import argparse
import copy
import math
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flowpacker'))

from datasets.cached_vf_dataset import CachedVFDataset, COND_DIM
from training.networks import FlowPrecond


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}


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
    torch.backends.cudnn.benchmark = True

    ds = CachedVFDataset(
        cache_dir=args.cache_dir,
        side_chain_path=args.side_chain_path,
        sequence_path=args.sequence_path,
        backbone_path=args.backbone_path,
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )

    # Student network with 27-dim cond
    net = FlowPrecond(
        in_channels=4,
        base_channels=128,
        x_channel_mult=[2, 4, 4, 2],
        emb_channel_mult=2,
        dropout=0.13,
        cond_dim=COND_DIM,
    ).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    total_kimg    = args.duration * 1000
    total_samples = total_kimg * 1000

    os.makedirs(args.outdir, exist_ok=True)
    run_dir = os.path.join(args.outdir, f'rcd-fm27-batch{args.batch}')
    os.makedirs(run_dir, exist_ok=True)
    print(f'Output: {run_dir}')

    log_file = open(os.path.join(run_dir, 'log.txt'), 'w')
    def log(msg):
        print(msg); log_file.write(msg+'\n'); log_file.flush()

    cur_nimg  = 0
    cur_tick  = 0
    tick_kimg = 50
    snap_ticks = 50
    start_time = time.time()
    tick_start = time.time()

    net.train()
    loader_iter = iter(loader)
    log(f'Training for {total_kimg} kimg, batch {args.batch}, cond_dim={COND_DIM}...')

    while cur_nimg < total_samples:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        xt       = batch['xt'].to(device)        # (B,512,4)
        vf       = batch['vf'].to(device)        # (B,512,4)
        t_raw    = batch['t'].to(device)         # (B,512)
        chi_mask = batch['chi_mask'].to(device)  # (B,512,4)
        cond     = batch['cond'].to(device)      # (B,512,27)

        B = xt.shape[0]

        # Flatten to valid residues only
        valid     = chi_mask.reshape(B*512, 4).any(dim=-1)
        xt_flat   = xt.reshape(B*512, 4)[valid].unsqueeze(1)   # (M,1,4)
        vf_flat   = vf.reshape(B*512, 4)[valid]                # (M,4)
        mask_flat = chi_mask.reshape(B*512, 4)[valid]          # (M,4)
        cond_flat = cond.reshape(B*512, COND_DIM)[valid]       # (M,27)
        t_flat    = t_raw.reshape(B*512)[valid]                # (M,)

        optimizer.zero_grad(set_to_none=True)

        pred_vf = net(xt_flat, cond_flat, t_flat).squeeze(1)   # (M,4)
        loss = ((pred_vf - vf_flat)**2 * mask_flat).sum(dim=-1).mean()
        # loss = ((pred_vf - x1_flat)**2 * mask_flat).sum(dim=-1).mean()

        loss.backward()
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # EMA
        ema_beta = 0.5 ** (args.batch / max(500*1000, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        cur_nimg += args.batch * 512
        done = cur_nimg >= total_samples
        kimg = cur_nimg / 1e3

        if cur_tick == 0 or kimg >= (cur_tick+1)*tick_kimg or done:
            tick_end = time.time()
            gpumem = torch.cuda.max_memory_allocated(device)/2**30
            torch.cuda.reset_peak_memory_stats()
            log(f"tick {cur_tick:<5d} kimg {kimg:<9.1f} loss {loss.item():<9.5f} "
                f"time {fmt_time(tick_end-start_time):<12s} "
                f"sec/tick {tick_end-tick_start:<7.1f} gpumem {gpumem:<6.2f}")
            tick_start = time.time()
            cur_tick += 1

            if cur_tick % snap_ticks == 0 or done:
                snap_path = os.path.join(run_dir, f'network-snapshot-{int(kimg):06d}.pkl')
                data = {'ema': copy.deepcopy(ema).cpu(), 'net': copy.deepcopy(net).cpu()}
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)
                log(f'  Saved {snap_path}')

        if done:
            break

    log('Training complete.')
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir',          default='rcd-runs-cached')
    parser.add_argument('--cache-dir',       default='datasets/vf_cache2')
    parser.add_argument('--side-chain-path', default='datasets/side_chain_data.npz')
    parser.add_argument('--sequence-path',   default='datasets/sequence_data.npz')
    parser.add_argument('--backbone-path',   default='datasets/backbone_data.npz')
    parser.add_argument('--duration',        type=int,   default=200)
    parser.add_argument('--batch',           type=int,   default=512)
    parser.add_argument('--lr',              type=float, default=8e-4)
    parser.add_argument('--seed',            type=int,   default=42)
    args = parser.parse_args()
    train(args)