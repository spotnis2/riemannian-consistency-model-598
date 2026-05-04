import copy
from copy import deepcopy

import torch

from training.manifolds import get_manifold


def _squeeze_mid_feat_dim(z: torch.Tensor) -> torch.Tensor:
    """[N, 1, F] -> [N, F]. Avoids N×N broadcast with chi_mask [N, F] in manifold ops."""
    if z.ndim == 3 and z.shape[1] == 1:
        return z.squeeze(1)
    return z


def _align_node_feats(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Coerce z to mask.shape ([N, F] PyG χ). Prevents [N,F]±[N,1,F] -> [N,N,F] OOM."""
    if mask.ndim != 2:
        return _squeeze_mid_feat_dim(z)
    while z.ndim > 2:
        z = z.squeeze(1)
    if z.shape != mask.shape and z.numel() == mask.numel():
        z = z.reshape(mask.shape)
    return z


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
        xt, vf = self.manifold.vecfield(n, x, t)
        xt, vf = _squeeze_mid_feat_dim(xt), _squeeze_mid_feat_dim(vf)
        pred_vf = net(xt, t)
        pred_vf = _squeeze_mid_feat_dim(pred_vf)
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

    def __call__(self, net, x, x_mask, cond, batch, iter_steps):
        def align(z):
            return _align_node_feats(z, x_mask)

        t = torch.rand(x.size(0), device=x.device) * self.tmax
        t_expand = t.view(-1, *([1] * (x.ndim - 1)))
        n = self.manifold.rand(*x.shape, device=x.device)
        n[x_mask == 0] = x[x_mask == 0]        #FREEZE INVALID POSITIONS
        xt, vf = self.manifold.vecfield(n, x, t)
        xt, vf = align(xt), align(vf)
        if self.distillation:
            with torch.no_grad():
                vf = self.teacher_model(t, xt, batch)
        vf = align(vf)
        vf = vf * x_mask
        # Here, we need to modify the tangent vector with the Jacobian to account for the potential coordinate transform.
        # For SO(3), the 3-vector representation is NOT the canonical Riemannian coordinate, so there will be a Jacobian term.
        # For Torus and Sphere, the ambient coordinates are Riemannian, so no Jacobian is needed (Jacobian is identity).
        tangents = (
            self.manifold.right_jac_inv(xt, vf) if hasattr(self.manifold, 'right_jac_inv') else vf,  # dx
            torch.zeros_like(cond),
            torch.ones_like(t)  # dt
        )

        # EDM2 modifies the parameters inplace, which will fail the forward-mode JVP calculation.
        # If you are not using EDM2, you may consider using torch.func.jvp for potentially better efficiency.
        pred_vf, dvf = torch.autograd.functional.jvp(net, (xt, cond, t), tangents, create_graph=True)
        # pred_vf, dvf = torch.func.jvp(net, (xt, t), tangents)
        pred_vf = align(pred_vf)
        dvf = align(dvf.detach())
        pred_vf_detach = pred_vf.detach()
        u = align((1 - t_expand) * pred_vf)
        pred_x1 = align(self.manifold.exp(xt, u))
        pred_x1_detach = pred_x1.detach()
        with torch.no_grad():
            # tangent warmup
            r = min(1.0, iter_steps + 1 / self.tangent_warmup_steps)
            cov_deriv = self.manifold.cov_deriv(pred_vf_detach, dvf, vf, xt).detach()
            cov_deriv = align(cov_deriv)
            du = align(-pred_vf_detach + (1 - t_expand) * cov_deriv * r)
            if not self.simplified:
                dexp_x = self.manifold.dexp_x(xt, u, du)
                dexp_u = self.manifold.dexp_u(xt, u, vf)
                g = align((dexp_x + dexp_u).detach())
            else:
                g = align((vf + du).detach())

        # tangent normalization (clip_jvp keeps trailing dims; align fixes any leftover [N,1,F])
        g_normed = align(clip_jvp(g, self.jvp_max_norm).detach())
        if not self.simplified:
            print(self.manifold.inner_with_mask(
                align(pred_x1_detach - pred_x1 + g_normed), g_normed, pred_x1_detach, x_mask
            ).shape)
            print((t / 1-t).shape)
            loss = self.manifold.inner_with_mask(
                align(pred_x1_detach - pred_x1 + g_normed), g_normed, pred_x1_detach, x_mask
            ) * (t / (1 - t)).square()
        else:
            loss = self.manifold.inner_with_mask(
                align(pred_vf.detach() - pred_vf + g_normed), g_normed, xt, x_mask
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
        t_expand = t.view(-1, *([1] * (x.ndim - 1)))
        n = self.manifold.rand(*x.shape, device=x.device)
        xt, vf = self.manifold.vecfield(n, x, t)
        xt, vf = _squeeze_mid_feat_dim(xt), _squeeze_mid_feat_dim(vf)
        if self.distillation:
            with torch.no_grad():
                vf = self.teacher_model(xt, t)
        vf = _squeeze_mid_feat_dim(vf)

        pred_x1 = self.manifold.exp(
            xt, (1 - t_expand) * _squeeze_mid_feat_dim(net(xt, t))
        )
        with torch.no_grad():
            xt_hat = self.manifold.exp(xt, self.dt * vf)
            pred_x1_hat = self.manifold.exp(
                xt_hat,
                (1 - t_expand - self.dt) * _squeeze_mid_feat_dim(net_clone(xt_hat, t + self.dt)),
            ).detach()
            del net_clone

        loss = self.manifold.norm2(
            self.manifold.log(pred_x1, pred_x1_hat), pred_x1.detach()
        ) / self.dt * (t / (1 - t)).unsqueeze(-1)
        return loss
