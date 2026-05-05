"""
evaluate_rcd.py — Evaluation for RCD flow matching model.
Uses 27-dim invariant conditioning and Euler integration sampling.

Usage:
    python evaluate_rcd.py \
        --checkpoint rcd-runs-cached/rcd-fm27-batch512/network-snapshot-200015.pkl
"""

import argparse
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F

COND_DIM = 27

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'X':'X','UNK':'X',
}
ONE_LETTER = 'ACDEFGHIKLMNPQRSTVWYX'
LETTER_TO_NUM = {c: i for i, c in enumerate(ONE_LETTER)}


def angular_diff_deg(pred, target):
    diff = pred - target
    diff = (diff + math.pi) % (2*math.pi) - math.pi
    return torch.abs(diff) * (180.0/math.pi)


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


def build_cond(backbone_path, sequence_path):
    """Build (N, 512, 27) invariant conditioning."""
    bb  = np.load(backbone_path)
    seq = np.load(sequence_path, allow_pickle=True)
    backbone = torch.tensor(bb['backbone'], dtype=torch.float32)  # (N,512,4,3)
    bb_mask  = torch.tensor(bb['mask'],     dtype=torch.float32)  # (N,512)
    names    = seq['names']
    N = backbone.shape[0]
    cond_list = []
    for i in range(N):
        bb_i=backbone[i]; msk_i=bb_mask[i]; nm_i=names[i]
        aa_num = torch.tensor(
            [LETTER_TO_NUM.get(THREE_TO_ONE.get(str(n),'X'),20) for n in nm_i],
            dtype=torch.long)
        aa_oh = F.one_hot(aa_num, num_classes=21).float()
        bbd = _get_bb_dihedral(bb_i[:,0], bb_i[:,1], bb_i[:,2])
        c = torch.cat([aa_oh, bbd.sin(), bbd.cos()], dim=-1) * msk_i.unsqueeze(-1)
        cond_list.append(c)
    return torch.stack(cond_list)  # (N,512,27)


@torch.no_grad()
def sample_euler(net, cond, device, n_steps=10):
    """Euler integration from t=1 (noise) to t=0 (clean). cond: (B,27)"""
    B = cond.shape[0]
    c = cond.to(device)
    x = torch.rand(B, 1, 4, device=device) * 2*math.pi  # uniform noise
    ts = torch.linspace(1.0, 0.0, n_steps+1, device=device)
    dt = ts[0] - ts[1]
    for i in range(n_steps):
        t = ts[i].expand(B)
        vf = net(x, c, t)           # (B,1,4)
        x = (x + dt * vf) % (2*math.pi)
    return x.squeeze(1)             # (B,4)


def evaluate(net, sc_path, cond, device, n_steps=10,
             test_split=0.2, batch_size=512, seed=42):
    sc     = np.load(sc_path)
    angles = torch.tensor(sc['side_chains'], dtype=torch.float32)  # (N,512,4) [-pi,pi]
    chi_valid  = (angles.abs() > 1e-6).float()   # (N,512,4)
    res_valid  = chi_valid.any(dim=-1).float()    # (N,512)

    N = angles.shape[0]
    n_test = max(1, int(N*test_split))
    torch.manual_seed(seed)
    test_idx = torch.randperm(N)[:n_test]

    angles_test   = angles[test_idx]       # (n_test,512,4) [-pi,pi]
    chi_mask_test = chi_valid[test_idx]
    res_mask_test = res_valid[test_idx]
    cond_test     = cond[test_idx]

    valid        = res_mask_test.bool()
    gt_flat      = angles_test[valid]      # (M,4) [-pi,pi]
    cond_flat    = cond_test[valid]        # (M,27)
    chimask_flat = chi_mask_test[valid]

    # Convert gt to [0,2pi] to match model output
    gt_wrapped = gt_flat % (2*math.pi)

    M = gt_flat.shape[0]
    print(f"Test: {n_test} proteins, {M} valid residues")

    net.eval()
    preds = []
    for start in range(0, M, batch_size):
        end  = min(start+batch_size, M)
        pred = sample_euler(net, cond_flat[start:end], device, n_steps=n_steps)
        preds.append(pred.cpu())
    pred_flat = torch.cat(preds, dim=0)  # (M,4) [0,2pi]

    diff_deg  = angular_diff_deg(pred_flat, gt_wrapped)
    valid_chi = chimask_flat.bool()
    all_diff  = diff_deg[valid_chi]
    mae_all   = all_diff.mean().item()
    acc_all   = (all_diff < 20.0).float().mean().item() * 100

    print(f"\n{'='*55}")
    print(f"Results ({n_steps}-step Euler)")
    print(f"{'='*55}")
    print(f"Overall  MAE: {mae_all:.2f}°   Accuracy: {acc_all:.2f}%")
    print(f"\nPer-chi:")
    print(f"  {'Chi':<8} {'MAE (°)':<12} {'Acc (%)':<12} N")
    print(f"  {'-'*45}")
    for i, name in enumerate(['chi1','chi2','chi3','chi4']):
        mi = valid_chi[:,i]
        if mi.sum() == 0:
            print(f"  {name:<8} {'N/A':<12} {'N/A':<12} 0")
            continue
        d = diff_deg[:,i][mi]
        print(f"  {name:<8} {d.mean().item():<12.2f} {(d<20).float().mean().item()*100:<12.2f} {mi.sum().item()}")
    print(f"{'='*55}")
    return {'mae': mae_all, 'acc': acc_all}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--sc-path',       default='datasets/side_chain_data.npz')
    parser.add_argument('--backbone-path', default='datasets/backbone_data.npz')
    parser.add_argument('--sequence-path', default='datasets/sequence_data.npz')
    parser.add_argument('--test-split',    type=float, default=0.2)
    parser.add_argument('--batch-size',    type=int,   default=512)
    parser.add_argument('--seed',          type=int,   default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        data = pickle.load(f)
    net = data['ema'].to(device).eval()
    print("Loaded.")

    print("Building conditioning vectors...")
    cond = build_cond(args.backbone_path, args.sequence_path)
    print(f"Cond: {cond.shape}")

    for n in [1, 5, 10, 20]:
        print(f"\n--- {n}-step Euler ---")
        evaluate(net, args.sc_path, cond, device,
                 n_steps=n, test_split=args.test_split,
                 batch_size=args.batch_size, seed=args.seed)


if __name__ == '__main__':
    main()