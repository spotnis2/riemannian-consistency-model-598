import copy
from copy import deepcopy

import torch
import math

from training.manifolds import get_manifold
from flowpacker.models.equiformer_v2.equiformer_v2 import PositionalEncodings
from flowpacker.dataset_cluster import get_edge_features

def clip_jvp(jvp: torch.Tensor, max_jvp_norm) -> torch.Tensor:
    if max_jvp_norm is None:
        return jvp
    jvp_norm = torch.linalg.vector_norm(jvp.reshape(jvp.shape[0], -1), dim=1)
    clip_coefficient = torch.clamp(max_jvp_norm / (jvp_norm + 1.e-6), max=1)
    return jvp * clip_coefficient.reshape(jvp.shape[0], *[1, ] * (len(jvp.shape) - 1))


class FlowLoss:
    def __init__(self, N, manifold, tmax=1.0):
        self.manifold = get_manifold(manifold, ndim=N)
        self.tmax = tmax

    def __call__(self, net, x):
        t = torch.rand(x.size(0), device=x.device) * self.tmax
        n = self.manifold.rand(*x.shape, device=x.device)
        xt, vf = self.manifold.vecfield(n, x, t[:, None])
        pred_vf = net(xt, t)
        diff = pred_vf - vf
        loss = self.manifold.inner(diff, diff, xt)
        return loss


class ConsistencyLoss:
    def __init__(
            self,
            N,
            manifold,
            simplified=False,
            tmax=0.995,
            tangent_warmup_steps=1,
            jvp_max_norm=10.0,
            distillation=False,
            teacher_model=None,
    ):
        if distillation:
            assert teacher_model is not None, 'Teacher model must be provided for distillation.'
        self.manifold = get_manifold(manifold, ndim=N)
        self.simplified = simplified
        self.tmax = tmax
        self.tangent_warmup_steps = tangent_warmup_steps
        self.jvp_max_norm = jvp_max_norm
        self.distillation = distillation
        self.teacher_model = teacher_model
        self.t_embedder = PositionalEncodings()

    def construct_gnn_node_features(self, batch, t, xt):
        node_feats = torch.cat([batch.aa_onehot, batch.bb_dihedral.sin(), batch.bb_dihedral.cos()], dim=-1) #[N, 27]
        t_for_embed = t.view(-1, 1) #[N, 1]
        t_embed = self.t_embedder(t_for_embed) #[N, 32]
        node_feats = torch.cat([t_embed, xt, node_feats], dim=-1) #[N, 32 + 4 + 27]
        return node_feats


    def __call__(self, net, x, x_mask, cond, batch, iter_steps):
   
        t = torch.rand(x.size(0), device=x.device) * self.tmax
        #i think these are in (0, 2pi), whereas x is in (-pi, pi)
        n = self.manifold.rand(*x.shape, device=x.device) 
        x_wrapped = (x + math.pi) % (2 * math.pi)
        n[x_mask == 0] = x_wrapped[x_mask == 0]        #FREEZE INVALID POSITIONS
        xt, vf = self.manifold.vecfield(n, x_wrapped, t) #is this properly handling the  torus?
        if self.distillation:
            with torch.no_grad():
                vf = self.teacher_model(t.unsqueeze(-1), xt, batch)
        vf = vf * x_mask #target vector field
        xt = xt * x_mask 
        # Here, we need to modify the tangent vector with the Jacobian to account for the potential coordinate transform.
        # For SO(3), the 3-vector representation is NOT the canonical Riemannian coordinate, so there will be a Jacobian term.
        # For Torus and Sphere, the ambient coordinates are Riemannian, so no Jacobian is needed (Jacobian is identity).
        # tangents = (
        #     self.manifold.right_jac_inv(xt, vf) if hasattr(self.manifold, 'right_jac_inv') else vf,  # dx
        #     torch.zeros_like(cond),
        #     torch.ones_like(t)  # dt
        # )
        dx = self.manifold.right_jac_inv(xt, vf) if hasattr(self.manifold, 'right_jac_inv') else vf
        dt = torch.ones_like(t)

        def f(xt, t):
            node_feats = self.construct_gnn_node_features(batch, t, xt) #[N, 32 + 4 + 27]
            #[2, num_residues*batch_size*k]
            #does not depend on xt or t. the tangent needs to be zero.
            edge_index = batch.edge_index #idk, I think in the flowpacker repo, they reconstruct this, but with virtual C_beta positions instead?
            # this is not dependent on xt or t. the tangent needs to be zero.
            edge_feats = get_edge_features(batch.pos, edge_index, None, False, None)  #[E(batch_size*num_residues*k), 65]
            return net(node_feats, edge_index, edge_feats)

        # EDM2 modifies the parameters inplace, which will fail the forward-mode JVP calculation.
        # If you are not using EDM2, you may consider using torch.func.jvp for potentially better efficiency.

        #CHANGE FOR GNN STUDENT
        #ensures that only xt and t are differentiated.
        pred_vf, dvf = torch.autograd.functional.jvp(f, (xt, t), (dx, dt), create_graph=True)
        #pred_vf, dvf = torch.autograd.functional.jvp(net, (xt, cond, t), tangents, create_graph=True)
        # pred_vf, dvf = torch.func.jvp(net, (xt, t), tangents)
       
        dvf = dvf.detach()
        pred_vf_detach = pred_vf.detach()
        u = (1 - t.unsqueeze(-1)) * pred_vf
        pred_x1 = self.manifold.exp(xt, u)
        pred_x1_detach = pred_x1.detach()
        with torch.no_grad():
            # tangent warmup
            r = min(1.0, iter_steps + 1 / self.tangent_warmup_steps)
            cov_deriv = self.manifold.cov_deriv(pred_vf_detach, dvf, vf, xt).detach()
            du = -pred_vf_detach + (1 - t.unsqueeze(-1)) * cov_deriv * r
            if not self.simplified:
                dexp_x = self.manifold.dexp_x(xt, u, du)
                dexp_u = self.manifold.dexp_u(xt, u, vf)
                g = (dexp_x + dexp_u).detach()
            else:
                g = (vf + du).detach()

        # tangent normalization (clip_jvp keeps trailing dims; align fixes any leftover [N,1,F])
        g_normed = clip_jvp(g, self.jvp_max_norm).detach()
        if not self.simplified:
            loss = self.manifold.inner_with_mask(
                (pred_x1_detach - pred_x1 + g_normed), g_normed, pred_x1_detach, x_mask
            ) * (t / (1 - t)).square()
        else:
            loss = self.manifold.inner_with_mask(
                (pred_vf.detach() - pred_vf + g_normed), g_normed, xt, x_mask
            ) * (t / (1 - t)).unsqueeze(-1).square()
        return loss

class DiscreteConsistencyLoss:
    def __init__(
            self,
            N,
            manifold,
            tmax=0.99,
            dt=0.01,
            distillation=False,
            teacher_model=None,

    ):
        if distillation:
            assert teacher_model is not None, 'Teacher model must be provided for distillation.'
        self.manifold = get_manifold(manifold, ndim=N)
        self.distillation = distillation
        self.teacher_model = teacher_model
        self.dt = dt
        self.tmax = tmax

    def __call__(self, net, x, iter_steps):
        net_clone = deepcopy(net)  # workaround for inplace modification in EDM2
        t = torch.rand(x.size(0), device=x.device) * self.tmax
        t_expand = t[:, None, None]
        n = self.manifold.rand(*x.shape, device=x.device)
        xt, vf = self.manifold.vecfield(n, x, t[:, None])
        if self.distillation:
            with torch.no_grad():
                vf = self.teacher_model(xt, t)

        pred_x1 = self.manifold.exp(xt, (1 - t_expand) * net(xt, t))
        with torch.no_grad():
            xt_hat = self.manifold.exp(xt, self.dt * vf)
            pred_x1_hat = self.manifold.exp(
                xt_hat, (1 - t_expand - self.dt) * net_clone(xt_hat, t + self.dt)
            ).detach()
            del net_clone

        loss = self.manifold.norm2(
            self.manifold.log(pred_x1, pred_x1_hat), pred_x1.detach()
        ) / self.dt * (t / (1 - t)).unsqueeze(-1)
        return loss
