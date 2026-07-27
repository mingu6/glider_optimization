import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from .cu_kulfan_airfoil import get_aero_from_kulfan_parameters_cuda

logger = logging.getLogger(__name__)


def mirror_full(y, c, xle, twist):
    """Mirror half-wing stations (root -> tip) into a full-span array, sorted by y."""
    y_full = np.concatenate((-y[::-1], y[1:]))
    c_full = np.concatenate((c[::-1], c[1:]))
    xle_full = np.concatenate((xle[::-1], xle[1:]))
    tw_full = np.concatenate((twist[::-1], twist[1:]))
    order = np.argsort(y_full)
    return y_full[order], c_full[order], xle_full[order], tw_full[order]


def segment_core(P, A, B, gamma=1.0, rc=0.01, eps_fac=0.05):
    """Biot-Savart velocity induced at P by a straight vortex segment A->B, with a
    Rankine viscous core of radius `rc` to regularize the self-influence singularity."""
    r1 = P - A
    r2 = P - B
    r0 = B - A

    seg_len = np.linalg.norm(r0)
    eps = eps_fac * seg_len + 1e-12
    r1n = max(np.linalg.norm(r1), eps)
    r2n = max(np.linalg.norm(r2), eps)
    cross = np.cross(r1, r2)
    cross2 = (np.dot(cross, cross)
              + (rc ** 2) * np.dot(r0, r0)
              + 0.25 * (rc ** 2) * (r1n * r1n + r2n * r2n))
    coeff = gamma / (4 * np.pi) * (np.dot(r0, (r1 / r1n - r2 / r2n)) / cross2)
    return coeff * cross


def trailing(P, A, B, A_w, B_w, gamma=1.0, rc=0.01):
    """Trailing-leg-only induced velocity (Trefftz plane / induced drag)."""
    return (segment_core(P, B, B_w, gamma, rc) +
            segment_core(P, A_w, A, gamma, rc))


def horseshoe(P, A, B, A_w, B_w, gamma=1.0, rc=0.01):
    """Full horseshoe vortex (bound leg A->B plus both trailing legs)."""
    return (segment_core(P, A, B, gamma, rc) +
            segment_core(P, B, B_w, gamma, rc) +
            segment_core(P, A_w, A, gamma, rc))


def build_llt_system(y_half, c_half, xle_half, twist_half, dihedral_deg=0.0):
    """Build the LLT influence matrices for the full (mirrored) wing.

    y_half, c_half, xle_half, twist_half : half-wing station coordinates (root -> tip).
    dihedral_deg : dihedral angle in degrees; sections are translated in z by
        y * tan(dihedral_deg), not rotated.

    Returns near-field (`D_nf`, full horseshoe) and Trefftz-plane (`D_tr`, trailing legs
    only) influence matrices mapping panel circulation to downwash at each control point,
    plus the panel geometry needed to integrate forces/moments over the span.
    """
    y_half = np.asarray(y_half, dtype=float)
    c_half = np.asarray(c_half, dtype=float)
    xle_half = np.asarray(xle_half, dtype=float)
    twist_half = np.asarray(twist_half, dtype=float)
    z_half = y_half * np.tan(np.deg2rad(float(dihedral_deg)))

    y, c, xle, twist = mirror_full(y_half, c_half, xle_half, twist_half)
    _, z, _, _ = mirror_full(y_half, z_half, xle_half, twist_half)

    vortex_loc = 0.25   # bound vortex at quarter-chord
    ctrl_loc = 0.25      # control point at quarter-chord

    n_pan = len(y) - 1
    yA, yB = y[:-1], y[1:]
    cA, cB = c[:-1], c[1:]
    xleA, xleB = xle[:-1], xle[1:]
    twA, twB = twist[:-1], twist[1:]
    zA, zB = z[:-1], z[1:]

    y_mid = 0.5 * (yA + yB)
    c_mid = 0.5 * (cA + cB)
    xle_mid = 0.5 * (xleA + xleB)
    tw_mid = 0.5 * (twA + twB)
    z_mid = 0.5 * (zA + zB)

    x_qA = xleA + vortex_loc * cA
    x_qB = xleB + vortex_loc * cB
    x_cp = xle_mid + ctrl_loc * c_mid

    # control points slightly below the bound-vortex line, as is standard for LLT/VLM
    CPts = np.column_stack([x_cp, y_mid, z_mid - 0.01 * c_mid])

    dy = np.abs(yB - yA)
    S = np.sum(0.5 * (cA + cB) * dy)
    cbar = np.sum(0.5 * (cA ** 2 + cB ** 2) * dy) / S

    A_q = np.column_stack([x_qA, yA, zA])
    B_q = np.column_stack([x_qB, yB, zB])

    # Per-panel sweep: horizontal-plane projection of the quarter-chord vortex segment,
    # cos(sweep) = |dy| / sqrt(dx_c4^2 + dy^2), which separates sweep from dihedral.
    dx_c4 = B_q[:, 0] - A_q[:, 0]
    dy_pan = np.abs(B_q[:, 1] - A_q[:, 1])
    cos_sweep = dy_pan / np.sqrt(dx_c4 ** 2 + dy_pan ** 2)
    cos2_sweep = cos_sweep ** 2

    # semi-infinite wake, approximated by a finite leg far downstream
    L_wake = 20.0 * max(c_mid.max(), 1.0)
    A_wq = A_q + np.array([L_wake, 0.0, 0.0])
    B_wq = B_q + np.array([L_wake, 0.0, 0.0])

    D_nf = np.zeros((n_pan, n_pan))
    D_tr = np.zeros((n_pan, n_pan))

    for i in range(n_pan):
        Pi = CPts[i]
        for j in range(n_pan):
            rc_nf = 0.25 * c_mid[j]
            rc_tr = 0.15 * c_mid[j]
            v_tr = trailing(Pi, A_q[j], B_q[j], A_wq[j], B_wq[j], rc=rc_tr)
            D_tr[i, j] = -v_tr[2]
            if i == j:
                # self-influence: bound leg singularity is regularized away, only the
                # trailing legs contribute (principal value)
                v_nf = (segment_core(Pi, B_q[j], B_wq[j], rc=rc_nf) +
                         segment_core(Pi, A_wq[j], A_q[j], rc=rc_nf))
            else:
                v_nf = horseshoe(Pi, A_q[j], B_q[j], A_wq[j], B_wq[j], rc=rc_nf)
            D_nf[i, j] = -v_nf[2]

    mirror_of = np.array([np.argmin(np.abs(y_mid + y_mid[i])) for i in range(n_pan)])

    return {
        "D_nf": D_nf, "D_tr": D_tr, "mirror_of": mirror_of,
        "c_mid": c_mid, "y_mid": y_mid, "tw_mid": tw_mid,
        "cbar": cbar, "S": S, "dy": dy, "span": float(y_half[-1] * 2.0),
        "cos_sweep": cos_sweep, "cos2_sweep": cos2_sweep,
    }


@dataclass
class LLTConst:
    dy: torch.Tensor
    c: torch.Tensor
    tw: torch.Tensor
    S: torch.Tensor
    cbar: torch.Tensor
    D_nf: torch.Tensor
    D_tr: torch.Tensor
    mirror_of: torch.Tensor
    cos_sweep: torch.Tensor
    cos2_sweep: torch.Tensor

    rho: torch.Tensor
    mu: torch.Tensor

    beta: float
    tol: float
    enforce_symmetry: bool
    model_size: str


def _local_flow(alpha, V, const):
    """Geometric AoA, sweep-corrected chord/Re at each panel, before induced-AoA correction."""
    tw = const.tw.unsqueeze(0)
    c = const.c.unsqueeze(0)
    cos_sw = const.cos_sweep.unsqueeze(0)
    V_n = V * cos_sw
    c_eff = c * cos_sw
    alpha_geo = alpha + tw
    Re = const.rho * V_n * c_eff / const.mu
    return alpha_geo, V_n, c_eff, Re


def eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, model_size: str) -> Dict[str, torch.Tensor]:
    """Evaluate the NeuralFoil surrogate at every (batch, panel) condition in one call.

    Supports a single global airfoil (`upper`/`lower` shape (8,)) broadcast to all panels,
    or a per-panel airfoil (shape (n_pan, 8)) for root-to-tip spanwise variation.
    """
    B, n_pan = alpha_eff.shape
    BN = B * n_pan
    alpha_flat = alpha_eff.reshape(-1)
    Re_flat = Re.reshape(-1)

    if upper.ndim == 1:
        upper_batch = upper.unsqueeze(0).expand(BN, -1)
        lower_batch = lower.unsqueeze(0).expand(BN, -1)
        LE_batch = LE.reshape(-1)[0].expand(BN)
        TE_batch = TE.reshape(-1)[0].expand(BN)
    elif upper.ndim == 2:
        if upper.shape[0] != n_pan or lower.shape[0] != n_pan:
            raise ValueError(
                f"per-panel Kulfan weights must have shape (n_pan={n_pan}, 8); "
                f"got upper {tuple(upper.shape)}, lower {tuple(lower.shape)}"
            )
        upper_batch = upper.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)
        lower_batch = lower.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)

        def to_panel(v):
            v = v.reshape(-1)
            if v.numel() == 1:
                return v[0].expand(n_pan)
            if v.numel() == n_pan:
                return v
            raise ValueError(f"per-panel LE/TE must be scalar or (n_pan,), got {tuple(v.shape)}")

        LE_batch = to_panel(LE).unsqueeze(0).expand(B, -1).reshape(BN)
        TE_batch = to_panel(TE).unsqueeze(0).expand(B, -1).reshape(BN)
    else:
        raise ValueError(f"upper must have ndim 1 (global airfoil) or 2 (per-panel), got {upper.ndim}")

    aero = get_aero_from_kulfan_parameters_cuda(
        {
            "upper_weights_cuda": upper_batch,
            "lower_weights_cuda": lower_batch,
            "leading_edge_weight_cuda": LE_batch,
            "TE_thickness_cuda": TE_batch,
        },
        alpha_flat,
        Re_flat,
        device=alpha_eff.device.type,
        model_size=model_size,
    )
    return {
        "CL": aero["CL"].reshape(B, n_pan),
        "CD": aero["CD"].reshape(B, n_pan),
        "CM": aero["CM"].reshape(B, n_pan),
        "analysis_confidence": aero["analysis_confidence"].reshape(B, n_pan),
    }


def _G(Gamma, alpha, V, upper, lower, LE, TE, const: LLTConst):
    """One Picard update Gamma -> Gamma_new (differentiable)."""
    w_nf = Gamma @ const.D_nf.T
    alpha_geo, V_n, c_eff, Re = _local_flow(alpha, V, const)
    alpha_eff = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V_n))

    aero = eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, const.model_size)
    Gamma_star = 0.5 * V_n * c_eff * aero["CL"]
    Gamma_new = (1.0 - const.beta) * Gamma + const.beta * Gamma_star

    if const.enforce_symmetry:
        Gamma_new = 0.5 * (Gamma_new + Gamma_new[:, const.mirror_of])
    return Gamma_new


def _F(Gamma, alpha, V, upper, lower, LE, TE, const: LLTConst):
    """Residual F(Gamma) = Gamma - G(Gamma); the LLT solution is its root."""
    return Gamma - _G(Gamma, alpha, V, upper, lower, LE, TE, const)


def _compute_coeffs(Gamma, alpha, V, upper, lower, LE, TE, const: LLTConst):
    """Integrate panel circulation/drag/moment into wing CL, CD, CM. Returns (B, 3)."""
    c = const.c.unsqueeze(0)
    dy = const.dy.unsqueeze(0)
    cos_sw = const.cos_sweep.unsqueeze(0)
    cos2_sw = const.cos2_sweep.unsqueeze(0)
    cos3_sw = cos_sw * cos2_sw

    w_tr = Gamma @ const.D_tr.T
    alpha_geo, V_n, c_eff, Re = _local_flow(alpha, V, const)
    w_nf = Gamma @ const.D_nf.T
    alpha_eff = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V_n))

    aero = eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, const.model_size)
    cd, cm = aero["CD"], aero["CM"]

    q = 0.5 * const.rho * (V ** 2)
    D_profile = q * c * cos2_sw * cd
    # induced drag from the Trefftz plane (momentum-conservative, per Munk's theorem)
    D_induced = const.rho * Gamma * w_tr

    L = const.rho * V.squeeze(-1) * torch.sum(Gamma * dy, dim=-1)
    D = torch.sum((D_profile + D_induced) * dy, dim=-1)
    denom = (q.squeeze(-1) * const.S).clamp_min(1e-30)
    CL = L / denom
    CD = D / denom

    # pitching moment about the wing quarter-chord line (NeuralFoil CM convention)
    M = torch.sum(q * (c ** 2) * cos3_sw * cm * dy, dim=-1)
    CM = M / (q.squeeze(-1) * const.S * const.cbar).clamp_min(1e-30)

    return torch.stack([CL, CD, CM], dim=-1)


class LLTImplicitFn(torch.autograd.Function):
    """Solves the LLT fixed point Gamma* = G(Gamma*) by Picard iteration, and
    differentiates through it via the implicit function theorem rather than
    backpropagating through the iteration itself."""

    @staticmethod
    def forward(
        ctx,
        alpha: torch.Tensor, V: torch.Tensor,
        upper: torch.Tensor, lower: torch.Tensor, LE: torch.Tensor, TE: torch.Tensor,
        dy: torch.Tensor, c: torch.Tensor, tw: torch.Tensor, S: torch.Tensor, cbar: torch.Tensor,
        D_nf: torch.Tensor, D_tr: torch.Tensor, mirror_of: torch.Tensor,
        cos_sweep: torch.Tensor, cos2_sweep: torch.Tensor,
        rho: torch.Tensor, mu: torch.Tensor,
        beta: float, tol: float, max_iter: int, enforce_symmetry: bool, model_size: str,
    ):
        const = LLTConst(
            dy=dy, c=c, tw=tw, S=S, cbar=cbar,
            D_nf=D_nf, D_tr=D_tr, mirror_of=mirror_of,
            cos_sweep=cos_sweep, cos2_sweep=cos2_sweep,
            rho=rho, mu=mu,
            beta=beta, tol=tol, enforce_symmetry=enforce_symmetry, model_size=model_size,
        )
        alpha = alpha.reshape(-1, 1)
        V = V.reshape(-1, 1)

        with torch.no_grad():
            alpha_geo, V_n, c_eff, Re = _local_flow(alpha, V, const)
            aero0 = eval_nf_batched(upper, lower, LE, TE, alpha_geo, Re, const.model_size)
            Gamma = 0.5 * V_n * c_eff * aero0["CL"]

            converged = False
            for it in range(max_iter):
                Gamma_new = _G(Gamma, alpha, V, upper, lower, LE, TE, const)
                rel_diff = (torch.max(torch.abs(Gamma_new - Gamma))
                            / torch.max(torch.abs(Gamma)).clamp_min(1.0)).item()
                Gamma = Gamma_new
                if rel_diff < tol:
                    converged = True
                    break
            if not converged:
                logger.warning("LLT did not converge in %d iterations (final rel. change %.2e)", max_iter, rel_diff)

            C = _compute_coeffs(Gamma, alpha, V, upper, lower, LE, TE, const)
            if not torch.isfinite(C).all():
                logger.error("LLT produced non-finite CL/CD/CM")

            alpha_geo, V_n, c_eff, Re = _local_flow(alpha, V, const)
            w_nf = Gamma @ const.D_nf.T
            alpha_eff_pan = (alpha_geo - torch.rad2deg(torch.atan2(w_nf, V_n))).detach()
            Re_pan = Re.detach()

        ctx.beta, ctx.tol, ctx.enforce_symmetry, ctx.model_size = beta, tol, enforce_symmetry, model_size
        ctx.save_for_backward(
            Gamma, alpha, V, upper, lower, LE, TE,
            dy, c, tw, S, cbar, D_nf, D_tr, mirror_of,
            cos_sweep, cos2_sweep, rho, mu,
        )
        return C, alpha_eff_pan, Re_pan

    @staticmethod
    def backward(ctx, grad_C, grad_alpha_eff_pan=None, grad_Re_pan=None):
        """Implicit backward: solve J^T lambda = dL/dGamma for the adjoint lambda
        (J = dF/dGamma, built via n_pan batched VJPs), then combine the direct and
        implicit (lambda^T dF/dp) gradient contributions w.r.t. the Kulfan parameters."""
        (Gamma, alpha, V, upper, lower, LE, TE,
         dy, c, tw, S, cbar, D_nf, D_tr, mirror_of,
         cos_sweep, cos2_sweep, rho, mu) = ctx.saved_tensors

        const = LLTConst(
            dy=dy, c=c, tw=tw, S=S, cbar=cbar,
            D_nf=D_nf, D_tr=D_tr, mirror_of=mirror_of,
            cos_sweep=cos_sweep, cos2_sweep=cos2_sweep,
            rho=rho, mu=mu,
            beta=ctx.beta, tol=ctx.tol, enforce_symmetry=ctx.enforce_symmetry, model_size=ctx.model_size,
        )

        with torch.enable_grad():
            Gamma = Gamma.detach().requires_grad_(True)
            grad_C = grad_C.reshape(Gamma.shape[0], 3)

            C = _compute_coeffs(Gamma, alpha, V, upper, lower, LE, TE, const)
            L = (C * grad_C).sum()
            rhs = torch.autograd.grad(L, Gamma, retain_graph=True)[0]

            B, n_pan = Gamma.shape
            F_all = _F(Gamma, alpha, V, upper, lower, LE, TE, const)
            J = torch.zeros(B, n_pan, n_pan, device=Gamma.device, dtype=Gamma.dtype)
            for i in range(n_pan):
                mask = torch.zeros_like(F_all)
                mask[:, i] = 1.0
                (row_i,) = torch.autograd.grad(F_all, Gamma, grad_outputs=mask, retain_graph=(i < n_pan - 1))
                J[:, i, :] = row_i

            lam = torch.linalg.solve(J.mT, rhs.unsqueeze(-1)).squeeze(-1).detach()

            grads_direct = torch.autograd.grad(L, (upper, lower, LE, TE), retain_graph=True, allow_unused=True)
            Fval = _F(Gamma, alpha, V, upper, lower, LE, TE, const)
            grads_implicit = torch.autograd.grad(Fval, (upper, lower, LE, TE), grad_outputs=lam, allow_unused=True)

            grads = [
                gd if gi is None else (gd - gi if gd is not None else -gi)
                for gd, gi in zip(grads_direct, grads_implicit)
            ]

        return (
            None, None,             # alpha, V
            *grads,                  # upper, lower, LE, TE
            None, None, None, None, None,  # dy, c, tw, S, cbar
            None, None, None,        # D_nf, D_tr, mirror_of
            None, None,               # cos_sweep, cos2_sweep
            None, None,                # rho, mu
            None, None, None, None, None,  # beta, tol, max_iter, enforce_symmetry, model_size
        )
