import torch
import math
import numpy as np
import argparse
import pickle
from flowpacker.dataset_cluster import get_dataloader, get_edge_features
from pathlib import Path
from flowpacker.models.equiformer_v2.equiformer_v2 import PositionalEncodings

def protein_graph_conditioning(batch):
        bb_dihedrals, pos, aa_onehot, aa_m = batch.bb_dihedral, batch.pos, batch.aa_onehot, batch.aa_mask.float()

        pos_flat = pos.reshape(pos.shape[0], -1)

        initial_cond = torch.cat([bb_dihedrals.sin(), bb_dihedrals.cos(), pos_flat, aa_onehot], dim=-1)

        num_graphs = int(batch.num_graphs)
        dev = initial_cond.device
        dty = initial_cond.dtype
        dfeat = initial_cond.size(-1)
        bidx = batch.batch

        #global mean pooling - have to do this becaause global_mean_pool will include all nodes including padding ones
        sums = torch.zeros(num_graphs, dfeat, device=dev, dtype=dty)
        counts = torch.zeros(num_graphs, device=dev, dtype=dty)
        sums.index_add_(0, bidx, initial_cond * aa_m.unsqueeze(-1))
        counts.index_add_(0, bidx, aa_m)
        mean_pool = sums / counts.clamp(min=1e-8).unsqueeze(-1)
        node_count = torch.log1p(counts).unsqueeze(-1)


        row, col = batch.edge_index
        edge_counts = torch.zeros(num_graphs, device=dev, dtype=dty)
        if row.numel() > 0:
            eb = bidx[row]
            valid = (aa_m[row] != 0) & (aa_m[col] != 0)
            if valid.any():
                edge_counts.index_add_(
                    0, eb[valid], torch.ones(int(valid.sum().item()), device=dev, dtype=dty)
                )
        edge_count = torch.log1p(edge_counts).unsqueeze(-1)

        cond_vector = torch.cat([mean_pool, node_count, edge_count], dim=-1)

 
        return cond_vector

def construct_gnn_node_features(batch, t, xt):
        t_embedder = PositionalEncodings()
        node_feats = torch.cat([batch.aa_onehot, batch.bb_dihedral.sin(), batch.bb_dihedral.cos()], dim=-1) #[N, 27]
        t_for_embed = t.view(-1, 1) #[N, 1]
        t_embed = t_embedder(t_for_embed) #[N, 32]
        node_feats = torch.cat([t_embed, xt, node_feats], dim=-1) #[N, 32 + 4 + 27]
        return node_feats

def torus_wrap(x):
    return x % (2 * math.pi)

def angular_diff_deg(pred, target):
    """
    Shortest angular difference in degrees, shape-preserving.
    pred, target in [0, 2π)
    """
    diff = pred - target
    diff = (diff + math.pi) % (2 * math.pi) - math.pi# wrap to (-π, π]
    return torch.abs(diff) * (180.0 / math.pi)

def sample_1step(net, device, batch, eps=0.05):
    B = batch.chi.shape[0]

    # RCM network expects [B, 1, chi_dim]

    x_noise = torch.rand(B, 4, device=device) * 2 * math.pi
    t = torch.full((B,), eps, device=device)

    node_feats = construct_gnn_node_features(batch, t, x_noise)
    edge_index = batch.edge_index #idk, I think in the flowpacker repo, they reconstruct this, but with virtual C_beta positions instead?
    # this is not dependent on xt or t. the tangent needs to be zero.
    edge_feats = get_edge_features(batch.pos, edge_index, None, False, None)

    out = net(node_feats, edge_index, edge_feats)

    direct = torus_wrap(out.squeeze(1))
    x_noise_2d = x_noise.squeeze(1)
    rcm = torus_wrap(x_noise_2d + (1.0 - eps) * out.squeeze(1))


def sample_nstep(net, device, batch, n_step=5, eps=0.05):

    B = batch.chi.shape[0]
    x_noise = torch.rand(B, 4, device=device) * 2 * math.pi
    t = torch.ones((B,), device=device)
    schedule = np.linspace(eps, 1.0, n_step+1)
    schedule = torch.tensor(schedule)

    node_feats = construct_gnn_node_features(batch, t, x_noise)
    edge_index = batch.edge_index #idk, I think in the flowpacker repo, they reconstruct this, but with virtual C_beta positions instead?
    # this is not dependent on xt or t. the tangent needs to be zero.
    edge_feats = get_edge_features(batch.pos, edge_index, None, False, None)
    for i in range(n_step):
        t = t.to(device).float()
        x1_pred = net(node_feats, edge_index, edge_feats)
        x1_pred = torus_wrap(x1_pred)
        if i < n_step - 1:
            t_next = schedule[i + 1].expand(B)
            noise = torch.rand_like(x1_pred) * 2 * math.pi
            # geodesic interp: x0 + t*(x1-x0) on torus
            delta = torus_wrap(x1_pred - noise)
            x = torus_wrap(noise.cpu() + t_next.view(B, 1, 1) * delta.cpu())
            t = t_next

    return torus_wrap(x1_pred.squeeze(1))


def evaluate(net, device, n_steps):
    #---Load data---
    train_dl, test_dl, _, _ = get_dataloader(batch_size=1)
    pred_list = []
    true_list = []
    chi_mask_list = []
    for batch in test_dl:
        batch = batch.to(device)
        if n_steps == 1:
            pred = sample_1step(net, device, batch)  #pred: [N, 4] where N = batch_size * 512(num of residues, invalid + valid, in each protein)
        else:
            pred = sample_nstep(net, device, batch, n_step=n_steps)
        x_wrapped = (batch.chi + math.pi) % (2 * math.pi)
        pred_list.append(pred.cpu())  #pred_list: [num_proteins, N, 4]
        true_list.append(x_wrapped.cpu())  #true_list: [num_proteins, N, 4]
        chi_mask_list.append(batch.chi_mask.cpu())  #chi_mask_list: [num_proteins, N, 4]
    pred_flat = torch.cat(pred_list, dim=0)  #pred_flat: [num_proteins * N, 4]
    true_flat = torch.cat(true_list, dim=0)  #true_flat: [num_proteins * N, 4]
    chi_mask_flat = torch.cat(chi_mask_list, dim=0)  #chi_mask_flat: [num_proteins * N, 4]
    chi_mask_flat = chi_mask_flat.bool()
    #metrics
    diff_deg = angular_diff_deg(pred_flat, true_flat)  #diff_deg: [num_proteins * N, 4]
    all_diff = diff_deg[chi_mask_flat]  #all_diff: [num_proteins * N, 4]
    mae_all = all_diff.mean().item()
    acc_all = (all_diff < 20).float().mean().item() * 100

    print(f"\nOverall (all chi angles):")
    print(f"  Angle MAE      : {mae_all:.2f}°")
    print(f"  Angle Accuracy : {acc_all:.2f}%  (within 20°)")

    print(f"\nPer-chi angle metrics:")
    for i in range(4):
        chi_diff = diff_deg[..., i][chi_mask_flat[..., i]]
        mae_chi = chi_diff.mean().item()
        acc_chi = (chi_diff < 20).float().mean().item() * 100
        print(f"  Chi{i+1} MAE      : {mae_chi:.2f}°")
        print(f"  Chi{i+1} Accuracy : {acc_chi:.2f}%  (within 20°)")

  # Save results
    results = {
        'mae_overall': mae_all,
        'acc_overall': acc_all,
        'per_chi_mae': [],
        'per_chi_acc': [],
        'n_steps': n_steps
    }
    for i in range(4):
        mask_i = chi_mask_flat[:, i].bool()
        if mask_i.sum() == 0:
            results['per_chi_mae'].append(None)
            results['per_chi_acc'].append(None)
        else:
            diff_i = diff_deg[:, i][mask_i]
            results['per_chi_mae'].append(diff_i.mean().item())
            results['per_chi_acc'].append((diff_i < 20.0).float().mean().item() * 100)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to network-snapshot-*.pkl')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.checkpoint, 'rb') as f:
        data = pickle.load(f)

    net = data['ema'].to(device).eval()

    print("\n\nRunning multi-step comparison...")
    for n in [1]:
        print(f"\n--- {n}-step sampling ---")
        evaluate(
        net= net,
        device= device,
        n_steps= n
        )

if __name__ == "__main__":
    main()

