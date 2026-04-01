
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List
from .cu_kulfan_airfoil import get_aero_from_kulfan_parameters_cuda

_MODEL_SIZE_TO_ID = {
    "xxsmall": 0,
    "xsmall": 1,
    "small": 2,
    "medium": 3,
    "large": 4,
    "xlarge": 5,
    "xxlarge": 6,
    "xxxlarge": 7,
}
_ID_TO_MODEL_SIZE = {v: k for k, v in _MODEL_SIZE_TO_ID.items()}

_DEVICE_TO_ID = {"cpu": 0, "cuda": 1, "mps": 2}
_ID_TO_DEVICE = {v: k for k, v in _DEVICE_TO_ID.items()}


def mirror_full(y, c, xle, twist):
    y_full   = np.concatenate((-y[::-1], y[1:]))
    c_full   = np.concatenate(( c[::-1],  c[1:]))
    xle_full = np.concatenate((xle[::-1], xle[1:]))
    tw_full  = np.concatenate((twist[::-1], twist[1:]))
    # sort by y
    o = np.argsort(y_full)
    return y_full[o], c_full[o], xle_full[o], tw_full[o]

def segment_core(P, A, B, gamma=1.0, rc=0.01, eps_fac=0.05):
    r1 = P - A
    r2 = P - B
    r0 = B - A

    seg_len = np.linalg.norm(r0)
    eps = eps_fac * seg_len + 1e-12
    r1n = max(np.linalg.norm(r1), eps)
    r2n = max(np.linalg.norm(r2), eps)
    cross = np.cross(r1, r2)
    cross2 = (np.dot(cross, cross)
              + (rc**2)*np.dot(r0, r0)
              + 0.25*(rc**2)*(r1n*r1n + r2n*r2n))
    coeff = gamma/(4*np.pi) * (np.dot(r0, (r1/r1n - r2/r2n)) / cross2)
    return coeff * cross

def trailing(P, A, B, A_w, B_w, gamma=1.0, rc=0.01):
    """
    Trailing-only contribution (for Trefftz & induced drag) with consistent orientation.
    """
    return (segment_core(P, B,   B_w, gamma, rc) +
            segment_core(P, A_w, A,   gamma, rc))

def horseshoe(P, A, B, A_w, B_w, gamma=1.0, rc=0.01):
    """
    Correct loop orientation:
      A -> B   (bound)
      B -> B_w (downstream trailing leg)
      A_w -> A (upstream trailing leg)  <-- note the order!
    """
    return (segment_core(P, A,   B,   gamma, rc) +
            segment_core(P, B,   B_w, gamma, rc) +
            segment_core(P, A_w, A,   gamma, rc))

def build_llt_system(y_half, c_half, xle_half, twist_half):
    
    y_half    = np.array(y_half, dtype=float)
    c_half    = np.array(c_half, dtype=float)
    xle_half  = np.array(xle_half, dtype=float)
    twist_half= np.array(twist_half, dtype=float)
    y, c, xle, twist = mirror_full(y_half, c_half, xle_half, twist_half)
    
    vortex_location = 0.25  # as fraction of local chord
    ctrl_point_location = 0.25  # as fraction of local chord

    n_st = len(y)
    n_pan = n_st - 1
    yA, yB = y[:-1], y[1:]
    cA, cB = c[:-1], c[1:]
    xleA, xleB = xle[:-1], xle[1:]
    twA, twB = twist[:-1], twist[1:]

    y_mid = 0.5*(yA + yB)
    c_mid = 0.5*(cA + cB)
    xle_mid = 0.5*(xleA + xleB)
    tw_mid = 0.5*(twA + twB)

    x_qA = xleA + vortex_location*cA
    x_qB = xleB + vortex_location*cB
    x_cp = xle_mid + ctrl_point_location*c_mid

    # Control points at 0.75 c, slightly below the surface
    CPts = np.column_stack([x_cp, y_mid, -0.01 * c_mid])

    dy = np.abs(yB - yA)
    S = np.sum(0.5*(cA + cB) * dy)

    # ── Vortex-core saturation guard ─────────────────────────────────────────
    # rc_nf = 0.25 * c_mid is the regularisation radius used for self-induction
    # (the diagonal of D_nf).  When dy < rc_nf the control point sits inside
    # its own core: the bound-vortex self-induction is suppressed to ≈ 0,
    # downwash correction at that panel is lost, and cl is evaluated at the
    # full geometric AoA — producing a spurious circulation spike at the root
    # and corrupting induced-drag and implicit-adjoint gradients.
    rc_nf_pan = 0.25 * c_mid
    saturated = dy < rc_nf_pan
    if saturated.any():
        bad_idx  = np.where(saturated)[0]
        worst    = bad_idx[np.argmax(rc_nf_pan[bad_idx] / dy[bad_idx])]
        import warnings
        warnings.warn(
            f"build_llt_system: {saturated.sum()} panel(s) have dy < rc_nf "
            f"(worst: panel {worst}, y={y_mid[worst]:.4f} m, "
            f"dy={dy[worst]*1e3:.1f} mm, rc_nf={rc_nf_pan[worst]*1e3:.1f} mm). "
            "Core saturation will suppress self-induction and produce a spurious "
            "circulation spike. Reduce panel count or use a half-cosine (Multhopp) "
            "distribution to widen the innermost panels.",
            stacklevel=2,
        )
    # ─────────────────────────────────────────────────────────────────────────

    # Quarter-chord positions per panel (midpoints)
    x_c4A = xleA + 0.25*cA
    x_c4B = xleB + 0.25*cB
    x_c4_mid = 0.5*(x_c4A + x_c4B)

    # Reference point: quarter-chord at y = 0 on the symmetry axis

    x_ref=0.019
    #x_ref=0.032 # from flow5
    z_ref = -0.002  # your geometry uses z=0 for the quarter-chord line

    # Mean aerodynamic chord (length) for coefficient normalization
    cbar = np.sum(0.5*(cA**2 + cB**2) * dy) / S

    A_q  = np.column_stack([x_qA, yA, np.zeros_like(yA)])
    B_q  = np.column_stack([x_qB, yB, np.zeros_like(yB)])

    # Wake from the back edge (0.75 c)
    Lwake = 20.0 * max(c_mid.max(), 1.0)
    A_wq = A_q + np.array([Lwake, 0.0, 0.0])
    B_wq = B_q + np.array([Lwake, 0.0, 0.0])
    
    D_nf = np.zeros((n_pan, n_pan))   # near-field (full horseshoe)
    D_tr = np.zeros((n_pan, n_pan))   # Trefftz (trailing only)

    for i in range(n_pan):
        Pi = CPts[i]
        for j in range(n_pan):
            rc_nf = 0.25 * c_mid[j]
            rc_tr = 0.15 * c_mid[j]
            v_tr = trailing(Pi, A_q[j], B_q[j], A_wq[j], B_wq[j], gamma=1.0, rc=rc_tr)
            D_tr[i, j] = -v_tr[2]
            if i == j:
                # self influence = trailing legs only (principal value)
                v_nf = ( segment_core(Pi, B_q[j], B_wq[j], gamma=1.0, rc=rc_nf) +
                    segment_core(Pi, A_wq[j], A_q[j], gamma=1.0, rc=rc_nf) )
            else:
                v_nf = horseshoe(Pi, A_q[j], B_q[j], A_wq[j], B_wq[j], gamma=1.0, rc=rc_nf)
            D_nf[i, j] = -v_nf[2]

    mirror_of = np.empty(n_pan, dtype=int)
    for i in range(n_pan):
        j = np.argmin(np.abs(y_mid + y_mid[i]))  # y_j ~ -y_i
        mirror_of[i] = j

    return {'D_nf':D_nf,'D_tr':D_tr,'mirror_of':mirror_of, 'c_mid':c_mid, 'y_mid':y_mid,
                      'cbar':cbar,'x_c4_mid':x_c4_mid, 'x_ref':x_ref, 'z_ref':z_ref, 'dy':dy, 'S':S,
                       'n_pan':n_pan, 'tw_mid':tw_mid, 'span': max(y_half)*2.0}



@dataclass
class LLTConst:
    # geometry / wing
    dy: torch.Tensor
    y: torch.Tensor
    c: torch.Tensor
    tw: torch.Tensor
    S: torch.Tensor
    cbar: torch.Tensor
    x_c4: torch.Tensor
    span: torch.Tensor
    D_nf: torch.Tensor
    D_tr: torch.Tensor
    mirror_of: torch.Tensor

    # flow
    rho: torch.Tensor
    mu: torch.Tensor

    # solver
    n_iter: int
    beta: float
    tol: float
    enforce_symmetry: bool

    # cuNF
    model_size: str

def _G(
    Gamma: torch.Tensor,     # (B, n_pan)
    alpha: torch.Tensor,     # (B, 1)
    V: torch.Tensor,         # (B, 1)
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    const: LLTConst,
    return_details: bool = False,
) -> torch.Tensor:
    """
    One Picard update step Gamma -> Gamma_new (differentiable).
    """
    tw = const.tw.unsqueeze(0)  # (1,n_pan)
    c = const.c.unsqueeze(0)

    w_nf = Gamma @ const.D_nf.T  # (B,n_pan)

    alpha_geo = alpha + tw
    alpha_eff = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V))

    Re = const.rho * V * c / const.mu

    aero = _eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, const)
    cl = aero["CL"]

    Gamma_star = 0.5 * V * c * cl
    Gamma_new = (1.0 - const.beta) * Gamma + const.beta * Gamma_star

    if const.enforce_symmetry:
        j = const.mirror_of
        Gamma_new = 0.5 * (Gamma_new + Gamma_new[:, j])

    if return_details:
        return Gamma_new, alpha_eff, Re, aero
    return Gamma_new

def _F(
    Gamma: torch.Tensor,
    alpha: torch.Tensor,
    V: torch.Tensor,
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    const: LLTConst,
) -> torch.Tensor:
    """
    Residual: F(Gamma)=Gamma - G(Gamma). Root is Gamma*.
    """
    return Gamma - _G(Gamma, alpha, V, upper, lower, LE, TE, const)

def _eval_nf_batched(
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    alpha_eff: torch.Tensor,  # (B, n_pan)
    Re: torch.Tensor,         # (B, n_pan)
    const: LLTConst,
) -> Dict[str, torch.Tensor]:
    """
    Single cuNeuralFoil call for (B, n_pan) by flattening to (B*n_pan,).
    Keeps gradient paths to upper/lower/LE/TE.
    """
    dev = alpha_eff.device
    dev_str = dev.type

    B, n_pan = alpha_eff.shape
    alpha_flat = alpha_eff.reshape(-1)
    Re_flat = Re.reshape(-1)
    BN = alpha_flat.numel()

    # expand (no data copy; gradients accumulate correctly)
    # Expand parameters to (BN, ...) for a single cuNeuralFoil call.
    # Supports either:
    #   - global airfoil: upper/lower (8,), LE/TE scalar or (1,)
    #   - per-panel airfoil: upper/lower (n_pan, 8), LE/TE (n_pan,) or scalar
    if upper.ndim == 1:
        # global airfoil -> broadcast to all panels
        upper_batch = upper.unsqueeze(0).expand(BN, -1)
        lower_batch = lower.unsqueeze(0).expand(BN, -1)

        LE0 = LE.reshape(-1)[0]
        TE0 = TE.reshape(-1)[0]
        LE_batch = LE0.expand(BN)
        TE_batch = TE0.expand(BN)

    elif upper.ndim == 2:
        if upper.shape[0] != n_pan or lower.shape[0] != n_pan:
            raise ValueError(
                f"Per-panel Kulfan must have shape (n_pan, 8); got upper {tuple(upper.shape)}, "
                f"lower {tuple(lower.shape)} with n_pan={n_pan}"
            )

        # (n_pan, 8) -> (B, n_pan, 8) -> (BN, 8)
        upper_batch = upper.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)
        lower_batch = lower.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)

        def _to_pan(v):
            v = v.reshape(-1)
            if v.numel() == 1:
                return v[0].expand(n_pan)
            if v.numel() == n_pan:
                return v
            raise ValueError(
                f"Per-panel LE/TE must be scalar or (n_pan,), got {tuple(v.shape)} with n_pan={n_pan}"
            )

        LE_pan = _to_pan(LE)
        TE_pan = _to_pan(TE)

        LE_batch = LE_pan.unsqueeze(0).expand(B, -1).reshape(BN)
        TE_batch = TE_pan.unsqueeze(0).expand(B, -1).reshape(BN)

    else:
        raise ValueError(f"Unsupported upper.ndim={upper.ndim}; expected 1 (global) or 2 (per-panel)")

    kulfan_batch = {
        "upper_weights_cuda": upper_batch,
        "lower_weights_cuda": lower_batch,
        "leading_edge_weight_cuda": LE_batch,
        "TE_thickness_cuda": TE_batch,
    }

    aero = get_aero_from_kulfan_parameters_cuda(
        kulfan_batch,
        alpha_flat, 
        Re_flat,
        device=dev_str,
        model_size=const.model_size,
    )

    # Hard assertion: if this fails, implicit gradients cannot work
    # if not aero["CL"].requires_grad:
    #     raise RuntimeError(
    #         "cuNeuralFoil output does not require grad. "
    #         "Implicit differentiation needs cuNeuralFoil to be differentiable on this backend."
    #     )

    return {
        "CL": aero["CL"].reshape(B, n_pan),
        "CD": aero["CD"].reshape(B, n_pan),
        "CM": aero["CM"].reshape(B, n_pan),
    }

def _compute_coeffs(
    Gamma: torch.Tensor,     # (B,n_pan)
    alpha: torch.Tensor,     # (B,1)
    V: torch.Tensor,         # (B,1)
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    const: LLTConst,
) -> torch.Tensor:
    """
    Returns (B,3): [CL, CD, CM_pitch]
    """
    c = const.c.unsqueeze(0)
    dy = const.dy.unsqueeze(0)
    tw = const.tw.unsqueeze(0)
    y = const.y.unsqueeze(0)
    x_c4 = const.x_c4.unsqueeze(0)

    # Trefftz induced velocity
    w_tr = Gamma @ const.D_tr.T
    w_nf= Gamma @ const.D_nf.T 

    alpha_geo = alpha + tw
    alpha_eff = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V))
    Re = const.rho * V * c / const.mu

    aero = _eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, const)
    cl = aero["CL"]
    cd = aero["CD"]
    cm = aero["CM"]

    q = 0.5 * const.rho * (V ** 2)  # (B,1)

    Lp = q * c * cl
    Dp = q * c * cd
    Di = const.rho * Gamma * w_tr #Treffz plane here for better momentum conservation - consistent with flow5 methodology

    # Integrals
    L = const.rho * V.squeeze(-1) * torch.sum(Gamma * dy, dim=-1)  # (B,)
    D = torch.sum((Dp + Di) * dy, dim=-1)

    denom = (q.squeeze(-1) * const.S).clamp_min(1e-30)
    CL = L / denom
    CD = D / denom

    # Pitching moment about the quarter-chord line (NeuralFoil CM is about c/4)
    Mprime_c4 = q * (c ** 2) * cm
    M_pitch = torch.sum(Mprime_c4 * dy, dim=-1)

    denom_pitch = (q.squeeze(-1) * const.S * const.cbar).clamp_min(1e-30)
    CM = M_pitch / denom_pitch

    return torch.stack([CL, CD, CM], dim=-1)  # (B,3)

class LLTImplicitFn(torch.autograd.Function):
    """
    Custom autograd Function:
      - forward converges Gamma* (no graph)
      - backward uses explicit Jacobian of F wrt Gamma per batch item
    """

    @staticmethod
    def forward(
        ctx,
        alpha: torch.Tensor,
        V: torch.Tensor,
        upper: torch.Tensor,
        lower: torch.Tensor,
        LE: torch.Tensor,
        TE: torch.Tensor,
        dy: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        tw: torch.Tensor,
        S: torch.Tensor,
        cbar: torch.Tensor,
        x_c4: torch.Tensor,
        span: torch.Tensor,
        D_nf: torch.Tensor,
        D_tr: torch.Tensor,
        mirror_of: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor,
        beta_t: torch.Tensor,
        tol_t: torch.Tensor,
        n_iter_t: torch.Tensor,
        max_iter_t: torch.Tensor,
        enforce_sym_t: torch.Tensor,
        model_size_id: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:

        model_size = _ID_TO_MODEL_SIZE[int(model_size_id.item())]
        ctx.model_size_id = int(model_size_id.item())
        ctx.device_id = int(device_id.item())

        const = LLTConst(
            dy=dy, y=y, c=c, tw=tw, S=S, cbar=cbar, x_c4=x_c4, span=span,
            D_nf=D_nf, D_tr=D_tr, mirror_of=mirror_of,
            rho=rho, mu=mu,
            n_iter=int(n_iter_t.item()),
            beta=float(beta_t.item()),
            tol=float(tol_t.item()),
            enforce_symmetry=bool(enforce_sym_t.item() > 0.5),
            model_size=model_size,
        )

        alpha2 = alpha.reshape(-1, 1)
        V2 = V.reshape(-1, 1)
        B = alpha2.shape[0]

        with torch.no_grad():
            # Initial guess via NF at alpha_geo
            tw0 = const.tw.unsqueeze(0)
            c0 = const.c.unsqueeze(0)
            alpha_geo = alpha2 + tw0
            Re = const.rho * V2 * c0 / const.mu

            aero0 = _eval_nf_batched(upper, lower, LE, TE, alpha_geo, Re, const)
            Gamma = 0.5 * V2 * c0 * aero0["CL"]

            max_iter = int(max_iter_t.item())
            n_iter = const.n_iter
            converged = False
            final_rel_diff = float('inf')
            residual_history = []  # 🔍 Track residual for gradient analysis
            
            for iter_idx in range(max_iter):
                Gamma_new = _G(Gamma, alpha2, V2, upper, lower, LE, TE, const)
                diff = torch.max(torch.abs(Gamma_new - Gamma))
                denom = torch.max(torch.tensor(1.0, device=Gamma.device), torch.max(torch.abs(Gamma)))
                rel_diff = (diff / denom).item()
                final_rel_diff = rel_diff
                
                # 🔍 Track residual history for gradient analysis
                residual_history.append(rel_diff)
                
                if rel_diff < const.tol:
                    Gamma = Gamma_new
                    converged = True
                    if iter_idx < n_iter:
                        print(f"🔍 LLT converged at iteration {iter_idx+1}/{n_iter}, rel_diff={rel_diff:.2e}")
                    else:
                        print(f"🔍 LLT converged at iteration {iter_idx+1}/{max_iter} (adaptive), rel_diff={rel_diff:.2e}")
                    break
                Gamma = Gamma_new
            
            # 🔍 Store convergence info for backward pass decision
            ctx.converged = converged
            ctx.actual_iters = iter_idx + 1 if converged else max_iter
            ctx.residual_history = residual_history
            
            # 🔍 DIAGNOSTIC: Analyze residual gradient to find optimal clipping point
            if len(residual_history) >= 10:
                # Compute recent gradient (last 5 iterations)
                recent_gradient = abs(residual_history[-1] - residual_history[-5]) / 5
                ctx.residual_gradient = recent_gradient
                
                # Also compute mid-range gradient (around iteration 15-20)
                if len(residual_history) >= 20:
                    mid_gradient = abs(residual_history[19] - residual_history[14]) / 5
                    print(f"🔍 Residual gradients: recent={recent_gradient:.2e}, mid (iter 15-20)={mid_gradient:.2e}")
            else:
                ctx.residual_gradient = float('inf')
            
            if not converged:
                # Find which samples didn't converge
                diff_per_sample = torch.max(torch.abs(Gamma_new - Gamma), dim=1)[0]
                worst_idx = torch.argmax(diff_per_sample).item()
                worst_alpha = alpha2[worst_idx, 0].item()
                worst_V = V2[worst_idx, 0].item()
                worst_Re = (const.rho * worst_V * const.c.mean() / const.mu).item()
                print(f"⚠️  LLT did NOT converge after {max_iter} iterations, final rel_diff={final_rel_diff:.2e}")
                print(f"    Worst sample: AoA={worst_alpha:.1f}°, V={worst_V:.2f} m/s, Re≈{worst_Re:.0f}")

            Gamma_star = Gamma
            C = _compute_coeffs(Gamma_star, alpha2, V2, upper, lower, LE, TE, const)
            
            # --- Panel-wise conditions used for NF (for confidence diagnostics) ---
            tw0 = const.tw.unsqueeze(0)           # (1, n_pan)
            c0  = const.c.unsqueeze(0)            # (1, n_pan)

            # induced normal velocity at panels (same as in _compute_coeffs)
            w_nf = Gamma_star @ const.D_nf.T      # (B, n_pan)

            alpha_geo = alpha2 + tw0              # (B, n_pan)
            alpha_eff_pan = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V2))  # (B, n_pan)
            Re_pan = const.rho * V2 * c0 / const.mu                            # (B, n_pan)

            # detach: confidence is a diagnostic / constraint input, not part of implicit adjoint
            alpha_eff_pan_out = alpha_eff_pan.detach()
            Re_pan_out = Re_pan.detach()

            # Store final residual for backward pass decision
            ctx.final_residual = final_rel_diff
            
            # DIAGNOSTIC: Check for NaN/Inf in coefficients
            if not torch.isfinite(C).all():
                print(f"🚨 LLT produced non-finite coefficients!")
                print(f"   C stats: min={C.min().item():.6f}, max={C.max().item():.6f}")
                print(f"   Gamma stats: min={Gamma_star.min().item():.6f}, max={Gamma_star.max().item():.6f}")

        ctx.save_for_backward(
            Gamma_star, alpha2, V2,
            upper, lower, LE, TE,
            dy, y, c, tw, S, cbar, x_c4, span, D_nf, D_tr, mirror_of, rho, mu,
            beta_t, tol_t, n_iter_t, max_iter_t, enforce_sym_t
        )
        return C, alpha_eff_pan_out, Re_pan_out  # (B,3)

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor, grad_alpha_eff_pan=None, grad_Re_pan=None):
        """
        Matrix-free implicit backward using GMRES on (dF/dGamma)^T lambda = dL/dGamma.

        Removes the dense Jacobian materialization:
            J = autograd.functional.jacobian(...)
        which costs ~O(n_pan) residual evaluations and is spiky/slow.

        Instead:
        - Define JT_mv(v) = J^T v via a VJP:
              JT_mv(v) = d/dGamma <F(Gamma), v>
        - Solve for lambda with GMRES using only JT_mv.
        """
        saved = ctx.saved_tensors
        (
            Gamma_star, alpha2, V2,
            upper, lower, LE, TE,
            dy, y, c, tw, S, cbar, x_c4, span, D_nf, D_tr, mirror_of, rho, mu,
            beta_t, tol_t, n_iter_t, max_iter_t, enforce_sym_t
        ) = saved

        model_size = _ID_TO_MODEL_SIZE[int(ctx.model_size_id)]
        const = LLTConst(
            dy=dy, y=y, c=c, tw=tw, S=S, cbar=cbar, x_c4=x_c4, span=span,
            D_nf=D_nf, D_tr=D_tr, mirror_of=mirror_of,
            rho=rho, mu=mu,
            n_iter=int(n_iter_t.item()),
            beta=float(beta_t.item()),
            tol=float(tol_t.item()),
            enforce_symmetry=bool(enforce_sym_t.item() > 0.5),
            model_size=model_size,
        )

        # -------------------------
        # Helper: GMRES (Givens)
        # -------------------------
        def gmres_solve(matvec, b, max_iter=40, tol=1e-6):
            """
            Solve A x = b using GMRES with Givens rotations.

            - Works on CPU/CUDA/MPS (no torch.linalg.solve)
            - matvec: callable(v) -> A v
            - b: (n,) tensor
            """
            n = b.numel()
            device = b.device
            dtype = b.dtype

            # x0 = 0
            x = torch.zeros_like(b)
            r0 = b - matvec(x)
            beta = torch.sqrt(torch.sum(r0 * r0))

            # If already converged
            if float(beta) < tol:
                return x

            # Krylov basis V and Hessenberg H
            V = []
            V.append(r0 / beta)

            H = torch.zeros((max_iter + 1, max_iter), device=device, dtype=dtype)
            cs = torch.zeros((max_iter,), device=device, dtype=dtype)  # cos
            sn = torch.zeros((max_iter,), device=device, dtype=dtype)  # sin

            # g is RHS in least squares problem
            g = torch.zeros((max_iter + 1,), device=device, dtype=dtype)
            g[0] = beta

            def apply_givens(h_col, k):
                # Apply previous Givens rotations to the new Hessenberg column
                for i in range(k):
                    temp = cs[i] * h_col[i] + sn[i] * h_col[i + 1]
                    h_col[i + 1] = -sn[i] * h_col[i] + cs[i] * h_col[i + 1]
                    h_col[i] = temp
                return h_col

            def make_givens(a, b):
                # Compute Givens rotation (c,s) that zeros b
                if float(b) == 0.0:
                    return torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
                r = torch.sqrt(a * a + b * b)
                c = a / r
                s = b / r
                return c, s

            k_final = -1
            for k in range(max_iter):
                # Arnoldi step
                w = matvec(V[k])

                # Modified Gram-Schmidt
                for j in range(k + 1):
                    H[j, k] = torch.sum(w * V[j])
                    w = w - H[j, k] * V[j]

                H[k + 1, k] = torch.sqrt(torch.sum(w * w))
                if float(H[k + 1, k]) != 0.0:
                    V.append(w / H[k + 1, k])
                else:
                    # happy breakdown
                    V.append(torch.zeros_like(w))

                # Apply previous Givens to this column
                h_col = H[:, k]
                h_col = apply_givens(h_col, k)

                # New Givens to zero H[k+1,k]
                c_k, s_k = make_givens(h_col[k], h_col[k + 1])
                cs[k] = c_k
                sn[k] = s_k

                # Apply to Hessenberg column
                temp = cs[k] * h_col[k] + sn[k] * h_col[k + 1]
                h_col[k + 1] = -sn[k] * h_col[k] + cs[k] * h_col[k + 1]
                h_col[k] = temp

                # Apply to g
                temp_g = cs[k] * g[k] + sn[k] * g[k + 1]
                g[k + 1] = -sn[k] * g[k] + cs[k] * g[k + 1]
                g[k] = temp_g

                # Residual norm is |g[k+1]|
                res = torch.abs(g[k + 1])
                if float(res) <= tol * float(beta):
                    k_final = k
                    break

            if k_final < 0:
                k_final = max_iter - 1

            # Solve upper triangular system R y = g (back substitution)
            m = k_final + 1
            R = H[:m, :m]
            y_ls = g[:m].clone()

            y = torch.zeros((m,), device=device, dtype=dtype)
            for i in range(m - 1, -1, -1):
                ssum = torch.sum(R[i, i + 1:m] * y[i + 1:m]) if i + 1 < m else torch.tensor(0.0, device=device, dtype=dtype)
                y[i] = (y_ls[i] - ssum) / R[i, i]

            # x = V_m @ y
            x = torch.zeros_like(b)
            for i in range(m):
                x = x + y[i] * V[i]

            return x

        # -------------------------
        # Main implicit backward
        # -------------------------
        with torch.enable_grad():
            Gamma = Gamma_star.detach().requires_grad_(True)
            grad_C = grad_C.reshape(Gamma.shape[0], 3)

            # Forward recompute coefficients (differentiable)
            C = _compute_coeffs(Gamma, alpha2, V2, upper, lower, LE, TE, const)
            L = (C * grad_C).sum()

            # RHS: dL/dGamma  (B, n_pan)
            rhs = torch.autograd.grad(L, Gamma, retain_graph=True, create_graph=False)[0]

            B, n_pan = Gamma.shape
            lambda_all = torch.zeros_like(Gamma)

            # For matrix-free VJP, we need F connected to Gamma with a graph.
            # We'll solve per batch item for simplicity (B is usually small).
            for b in range(B):
                gb = Gamma[b:b+1]  # (1, n_pan)

                def JT_mv(v: torch.Tensor) -> torch.Tensor:
                    """
                    Compute (dF/dGamma)^T v for this batch item using a VJP.
                    v: (n_pan,)
                    returns: (n_pan,)
                    """
                    v = v.reshape(1, n_pan)

                    F_b = _F(
                        gb,
                        alpha2[b:b+1],
                        V2[b:b+1],
                        upper, lower, LE, TE,
                        const,
                    )  # (1, n_pan)

                    # VJP: d/dGamma <F, v>
                    JT_v = torch.autograd.grad(
                        outputs=F_b,
                        inputs=gb,
                        grad_outputs=v,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=False,
                    )[0]  # (1, n_pan)

                    return JT_v.reshape(n_pan)

                bvec = rhs[b].detach()
                lam_b = gmres_solve(JT_mv, bvec, max_iter=min(40, n_pan), tol=1e-6)
                lambda_all[b] = lam_b

            # We do NOT want gradients flowing through GMRES internals
            lambda_all = lambda_all.detach()

            # Direct term: dL/dp (Kulfan only)
            grads_direct = torch.autograd.grad(
                L, (upper, lower, LE, TE),
                retain_graph=True, allow_unused=True, create_graph=False
            )

            # Implicit term: lambda^T dF/dp
            Fval = _F(Gamma, alpha2, V2, upper, lower, LE, TE, const)

            if not Fval.requires_grad:
                raise RuntimeError(
                    "Fval does not require grad in implicit backward. "
                    "This means F is not connected to Kulfan params in autograd."
                )

            grads_impl = torch.autograd.grad(
                Fval, (upper, lower, LE, TE),
                grad_outputs=lambda_all,
                retain_graph=False, allow_unused=True, create_graph=False
            )

            # Combine: dL/dp = direct - implicit
            out_grads = []
            for gd, gi in zip(grads_direct, grads_impl):
                if gd is None and gi is None:
                    out_grads.append(None)
                elif gd is None:
                    out_grads.append(-gi)
                elif gi is None:
                    out_grads.append(gd)
                else:
                    out_grads.append(gd - gi)

            g_upper, g_lower, g_LE, g_TE = out_grads

        # Return grads aligned with forward() inputs of LLTImplicitFn.apply(...)
        # Updated to include max_iter_t parameter (one more None)
        return (
            None,  # alpha
            None,  # V
            g_upper,
            g_lower,
            g_LE,
            g_TE,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def llt_const_from_config(nf_cfg, device: torch.device) -> LLTConst:
    """
    Build an ``LLTConst`` from a ``NeuralFoilSamplingConfig`` instance.

    All geometry, physical constants and solver settings are read from the
    config, so the caller does not need to pass them separately.
    """
    comp = build_llt_system(
        nf_cfg.llt_y_half,
        nf_cfg.llt_c_half,
        nf_cfg.llt_xle_half,
        nf_cfg.llt_twist_half,
    )

    model_size = nf_cfg.llt_model_size or nf_cfg.neuralFoil_size

    def _t(val, dtype=torch.float32):
        return torch.as_tensor(val, dtype=dtype, device=device)

    return LLTConst(
        dy=_t(comp["dy"]),
        y=_t(comp["y_mid"]),
        c=_t(comp["c_mid"]),
        tw=_t(comp["tw_mid"]),
        S=_t(comp["S"]),
        cbar=_t(comp["cbar"]),
        x_c4=_t(comp["x_c4_mid"]),
        span=_t(comp["span"]),
        D_nf=_t(comp["D_nf"]),
        D_tr=_t(comp["D_tr"]),
        mirror_of=_t(comp["mirror_of"], dtype=torch.long),
        rho=_t(nf_cfg.llt_rho_air),
        mu=_t(nf_cfg.llt_mu_air),
        n_iter=nf_cfg.llt_n_iter,
        beta=nf_cfg.llt_beta,
        tol=nf_cfg.llt_tol,
        enforce_symmetry=nf_cfg.llt_enforce_symmetry,
        model_size=model_size,
    )


def run_llt(
    alpha: torch.Tensor,
    V: torch.Tensor,
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    const: LLTConst,
    max_iter: int = 200,
) -> torch.Tensor:
    """
    Run the LLT solver and return wing coefficients ``(B, 3)`` = [CL, CD, CM].

    All constant geometry/physics/solver parameters are bundled in ``const``
    (built via :func:`llt_const_from_config`).  The only variable inputs are
    the aerodynamic operating conditions and the Kulfan airfoil parameters.

    Args:
        alpha:    Angle of attack, shape ``(B,)`` or ``(B, 1)``, degrees.
        V:        Free-stream velocity, shape ``(B,)`` or ``(B, 1)``, m/s.
        upper:    Kulfan upper weights, shape ``(8,)`` or ``(n_pan, 8)``.
        lower:    Kulfan lower weights, shape ``(8,)`` or ``(n_pan, 8)``.
        LE:       Leading-edge weight, scalar or shape ``(n_pan,)``.
        TE:       TE thickness, scalar or shape ``(n_pan,)``.
        const:    Pre-built :class:`LLTConst` (use :func:`llt_const_from_config`).
        max_iter: Hard cap on Picard iterations (passed as ``max_iter_t``).

    Returns:
        Tensor of shape ``(B, 3)`` containing [CL, CD, CM] for each sample.
    """
    dev = alpha.device
    _s = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=dev)

    C, _, _ = LLTImplicitFn.apply(
        alpha,
        V,
        upper,
        lower,
        LE,
        TE,
        const.dy,
        const.y,
        const.c,
        const.tw,
        const.S,
        const.cbar,
        const.x_c4,
        const.span,
        const.D_nf,
        const.D_tr,
        const.mirror_of,
        const.rho,
        const.mu,
        _s(const.beta),
        _s(const.tol),
        _s(const.n_iter, torch.int64),
        _s(max_iter, torch.int64),
        _s(1.0 if const.enforce_symmetry else 0.0),
        _s(_MODEL_SIZE_TO_ID[const.model_size], torch.int64),
        _s(0, torch.int64),  # device_id (unused at runtime, kept for signature compat)
    )
    return C
