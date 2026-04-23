
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

def build_llt_system(y_half, c_half, xle_half, twist_half, z_half=None, dihedral_deg=0.0):
    """
    Build the LLT influence matrices for the full wing.

    Parameters
    ----------
    y_half, c_half, xle_half, twist_half : array-like
        Half-wing station coordinates (root → tip).
    z_half : array-like, optional
        Vertical z-positions of the half-wing stations (root → tip, in metres).
        If None, computed from ``dihedral_deg`` as ``y_half * tan(dihedral_deg)``.
    dihedral_deg : float
        Dihedral angle in degrees.  Used only when ``z_half`` is None.
        Γ = 0 (default) → flat wing → identical to previous behaviour.
    """
    y_half    = np.array(y_half, dtype=float)
    c_half    = np.array(c_half, dtype=float)
    xle_half  = np.array(xle_half, dtype=float)
    twist_half= np.array(twist_half, dtype=float)

    # --- z-coordinates for dihedral (Model B: sections translate, don't rotate) ---
    if z_half is None:
        z_half = y_half * np.tan(np.deg2rad(float(dihedral_deg)))
    z_half = np.array(z_half, dtype=float)

    # Mirror z symmetrically (both half-wings go UP with dihedral: z(-y) = z(|y|))
    y_full_unsorted = np.concatenate((-y_half[::-1], y_half[1:]))
    z_full_unsorted = np.concatenate(( z_half[::-1], z_half[1:]))
    sort_order = np.argsort(y_full_unsorted)
    z_full = z_full_unsorted[sort_order]

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

    # Panel z-coordinates (from dihedral)
    zA = z_full[:-1]
    zB = z_full[1:]
    z_mid = 0.5 * (zA + zB)

    x_qA = xleA + vortex_location*cA
    x_qB = xleB + vortex_location*cB
    x_cp = xle_mid + ctrl_point_location*c_mid

    # Control points at 0.25 c, slightly below the surface (z_mid offsets for dihedral)
    CPts = np.column_stack([x_cp, y_mid, z_mid - 0.01 * c_mid])

    dy = np.abs(yB - yA)
    S = np.sum(0.5*(cA + cB) * dy)

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

    A_q  = np.column_stack([x_qA, yA, zA])
    B_q  = np.column_stack([x_qB, yB, zB])

    # Per-panel sweep: horizontal-plane projection of the ¼-chord vortex segment.
    # cos(Λ_i) = |Δy| / sqrt(Δx_c4² + Δy²)  — separates sweep from dihedral z.
    _dx_c4     = B_q[:, 0] - A_q[:, 0]
    _dy_pan    = np.abs(B_q[:, 1] - A_q[:, 1])
    cos_sweep  = _dy_pan / np.sqrt(_dx_c4**2 + _dy_pan**2)  # (n_pan,)
    cos2_sweep = cos_sweep ** 2                              # (n_pan,)

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
                       'n_pan':n_pan, 'tw_mid':tw_mid, 'span': max(y_half)*2.0,
                       'cos_sweep': cos_sweep, 'cos2_sweep': cos2_sweep}



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
    cos_sweep: torch.Tensor   # (n_pan,)  cos(Λ) per panel, horizontal-plane sweep
    cos2_sweep: torch.Tensor  # (n_pan,)  cos²(Λ)

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

    cos_sw  = const.cos_sweep.unsqueeze(0)   # (1, n_pan)
    cos2_sw = const.cos2_sweep.unsqueeze(0)  # (1, n_pan)
    V_n     = V * cos_sw                     # (B, n_pan)  V·cosΛ
    c_eff   = c * cos_sw                     # (1, n_pan)  c·cosΛ  (section chord normal to LE)

    alpha_geo = alpha + tw
    alpha_eff = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V_n))

    Re = const.rho * V_n * c_eff / const.mu  # ρVc·cos²Λ/μ

    aero = _eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, const)
    cl = aero["CL"]

    Gamma_star = 0.5 * V_n * c_eff * cl      # ½Vc·cos²Λ·Cl
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

    cos_sw  = const.cos_sweep.unsqueeze(0)   # (1, n_pan)
    cos2_sw = const.cos2_sweep.unsqueeze(0)  # (1, n_pan)
    cos3_sw = cos_sw * cos2_sw               # (1, n_pan)  cos³(Λ) for moment
    V_n     = V * cos_sw                     # (B, n_pan)  V·cosΛ
    c_eff   = c * cos_sw                     # (1, n_pan)  c·cosΛ  (section chord normal to LE)

    alpha_geo = alpha + tw
    alpha_eff = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V_n))
    Re = const.rho * V_n * c_eff / const.mu  # ρVc·cos²Λ/μ

    aero = _eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, const)
    # cl = aero["CL"]
    cd = aero["CD"]
    cm = aero["CM"]

    q = 0.5 * const.rho * (V ** 2)  # (B,1)

    # Lp = q * c * cl # Local lift per unit span (before integrating with dy), unused in final CL but useful for debugging and understanding the distribution of lift along the span.
    Dp = q * c * cos2_sw * cd   # q·c·cos²Λ·Cd  (d_n/dy = 1/cosΛ cancels one power)
    Di = const.rho * Gamma * w_tr #Treffz plane here for better momentum conservation - consistent with flow5 methodology

    # Integrals
    L = const.rho * V.squeeze(-1) * torch.sum(Gamma * dy, dim=-1)  # (B,)
    D = torch.sum((Dp + Di) * dy, dim=-1)

    denom = (q.squeeze(-1) * const.S).clamp_min(1e-30)
    CL = L / denom
    CD = D / denom

    # Pitching moment about the quarter-chord line (NeuralFoil CM is about c/4)
    Mprime_c4 = q * (c ** 2) * cos3_sw * cm
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
        cos_sweep: torch.Tensor,
        cos2_sweep: torch.Tensor,
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
            cos_sweep=cos_sweep, cos2_sweep=cos2_sweep,
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
            # Initial guess via NF at alpha_geo (sweep-corrected)
            tw0       = const.tw.unsqueeze(0)
            c0        = const.c.unsqueeze(0)
            cos_sw0   = const.cos_sweep.unsqueeze(0)
            cos2_sw0  = const.cos2_sweep.unsqueeze(0)
            V_n0      = V2 * cos_sw0
            c_eff0    = c0 * cos_sw0   # c·cosΛ
            alpha_geo = alpha2 + tw0
            Re = const.rho * V_n0 * c_eff0 / const.mu  # ρVc·cos²Λ/μ

            aero0 = _eval_nf_batched(upper, lower, LE, TE, alpha_geo, Re, const)
            Gamma = 0.5 * V_n0 * c_eff0 * aero0["CL"]  # ½Vc·cos²Λ·Cl

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
            tw0        = const.tw.unsqueeze(0)             # (1, n_pan)
            c0         = const.c.unsqueeze(0)              # (1, n_pan)
            cos_sw_d   = const.cos_sweep.unsqueeze(0)      # (1, n_pan)
            cos2_sw_d  = const.cos2_sweep.unsqueeze(0)     # (1, n_pan)
            V_n_d      = V2 * cos_sw_d                     # (B, n_pan)
            c_eff_d    = c0 * cos_sw_d                     # (1, n_pan)  c·cosΛ

            # induced normal velocity at panels (same as in _compute_coeffs)
            w_nf = Gamma_star @ const.D_nf.T              # (B, n_pan)

            alpha_geo     = alpha2 + tw0                                                       # (B, n_pan)
            alpha_eff_pan = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V_n_d))               # (B, n_pan)
            Re_pan        = const.rho * V_n_d * c_eff_d / const.mu                            # (B, n_pan)

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
            dy, y, c, tw, S, cbar, x_c4, span, D_nf, D_tr, mirror_of,
            cos_sweep, cos2_sweep,
            rho, mu,
            beta_t, tol_t, n_iter_t, max_iter_t, enforce_sym_t
        )
        return C, alpha_eff_pan_out, Re_pan_out  # (B,3)

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor, grad_alpha_eff_pan=None, grad_Re_pan=None):
        """
        Batched implicit backward via explicit Jacobian + torch.linalg.solve.

        Builds the (B, n_pan, n_pan) block-diagonal Jacobian dF/dGamma with
        exactly n_pan full-batch VJP passes (one per row), then solves
        J^T lambda = dL/dGamma for all B items simultaneously via batched LU.

        Cost: n_pan NF calls (vs. up to 40*B in the old per-item GMRES loop).
        """
        saved = ctx.saved_tensors
        (
            Gamma_star, alpha2, V2,
            upper, lower, LE, TE,
            dy, y, c, tw, S, cbar, x_c4, span, D_nf, D_tr, mirror_of,
            cos_sweep, cos2_sweep,
            rho, mu,
            beta_t, tol_t, n_iter_t, max_iter_t, enforce_sym_t
        ) = saved

        model_size = _ID_TO_MODEL_SIZE[int(ctx.model_size_id)]
        const = LLTConst(
            dy=dy, y=y, c=c, tw=tw, S=S, cbar=cbar, x_c4=x_c4, span=span,
            D_nf=D_nf, D_tr=D_tr, mirror_of=mirror_of,
            cos_sweep=cos_sweep, cos2_sweep=cos2_sweep,
            rho=rho, mu=mu,
            n_iter=int(n_iter_t.item()),
            beta=float(beta_t.item()),
            tol=float(tol_t.item()),
            enforce_symmetry=bool(enforce_sym_t.item() > 0.5),
            model_size=model_size,
        )

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

            # Build (B, n_pan, n_pan) Jacobian dF/dGamma with n_pan full-batch VJP
            # passes. F[b, i] depends only on Gamma[b, :] (no cross-batch coupling),
            # so one VJP with mask[:, i]=1 gives row i for all B items at once.
            F_all = _F(Gamma, alpha2, V2, upper, lower, LE, TE, const)  # (B, n_pan)
            J = torch.zeros(B, n_pan, n_pan, device=Gamma.device, dtype=Gamma.dtype)
            for i in range(n_pan):
                mask = torch.zeros_like(F_all)
                mask[:, i] = 1.0
                (row_i,) = torch.autograd.grad(
                    F_all, Gamma,
                    grad_outputs=mask,
                    retain_graph=(i < n_pan - 1),
                    create_graph=False,
                )
                J[:, i, :] = row_i  # J[b, i, j] = dF[b,i]/dGamma[b,j]

            # Solve J^T lambda = rhs for all B items in one batched LAPACK call
            lambda_all = torch.linalg.solve(
                J.mT,               # (B, n_pan, n_pan)
                rhs.unsqueeze(-1),  # (B, n_pan, 1)
            ).squeeze(-1)           # (B, n_pan)

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
        return (
            None,  # alpha
            None,  # V
            g_upper,
            g_lower,
            g_LE,
            g_TE,
            None, None, None, None, None, None, None, None,  # dy, y, c, tw, S, cbar, x_c4, span
            None, None, None,                                 # D_nf, D_tr, mirror_of
            None, None,                                       # cos_sweep, cos2_sweep
            None, None,                                       # rho, mu
            None, None, None, None, None,                     # beta_t, tol_t, n_iter_t, max_iter_t, enforce_sym_t
            None, None,                                       # model_size_id, device_id
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

