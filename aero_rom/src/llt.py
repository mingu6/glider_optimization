# lifting-line (3D aggregation)

import numpy as np
import src.geometry as geom
import aerosandbox as asb
import pandas as pd
import torch
import cuneuralfoil
from cuneuralfoil.main import get_aero_from_kulfan_parameters_cuda

# --- MPS fix: ensure _ln_eps is float32, not float64 ---
if hasattr(cuneuralfoil.main, "_ln_eps"):
    cuneuralfoil.main._ln_eps = cuneuralfoil.main._ln_eps.to(dtype=torch.float32)

def v_segment_core(P, A, B, gamma=1.0, rc=0.01, eps_fac=0.05):
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

def v_horseshoe(P, A, B, A_w, B_w, gamma=1.0, rc=0.01):
    """
    Correct loop orientation:
      A -> B   (bound)
      B -> B_w (downstream trailing leg)
      A_w -> A (upstream trailing leg)  <-- note the order!
    """
    return (v_segment_core(P, A,   B,   gamma, rc) +
            v_segment_core(P, B,   B_w, gamma, rc) +
            v_segment_core(P, A_w, A,   gamma, rc))

# Trefftz-plane trailing-only operator (for induced drag)
def v_trailing(P, A, B, A_w, B_w, gamma=1.0, rc=0.01):
    """
    Trailing-only contribution (for Trefftz & induced drag) with consistent orientation.
    """
    return (v_segment_core(P, B,   B_w, gamma, rc) +
            v_segment_core(P, A_w, A,   gamma, rc))

def LLT_computational_params(y_half, c_half, xle_half, twist_half, airfoil_name):
    
    airfoil_CST=asb.Airfoil(geom.normalize_airfoil_name(airfoil_name)).to_kulfan_airfoil()
    y_half    = np.array(y_half, dtype=float)
    c_half    = np.array(c_half, dtype=float)
    xle_half  = np.array(xle_half, dtype=float)
    twist_half= np.array(twist_half, dtype=float)
    y, c, xle, twist = geom.mirror_full(y_half, c_half, xle_half, twist_half)
    
    """
    Build panels & control geometry (Weissinger L method)
    """
    vortex_location = 0.75  # as fraction of local chord
    ctrl_point_location = 0.75  # as fraction of local chord

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


    """
    Build downwash influence matrices for Weissinger L method.
    vortex_location, ctrl_point_location: fraction of local chord (0=LE, 1=TE)
    """
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
            v_tr = v_trailing(Pi, A_q[j], B_q[j], A_wq[j], B_wq[j], gamma=1.0, rc=rc_tr)
            D_tr[i, j] = -v_tr[2]
            if i == j:
                # self influence = trailing legs only (principal value)
                v_nf = ( v_segment_core(Pi, B_q[j], B_wq[j], gamma=1.0, rc=rc_nf) +
                    v_segment_core(Pi, A_wq[j], A_q[j], gamma=1.0, rc=rc_nf) )
            else:
                v_nf = v_horseshoe(Pi, A_q[j], B_q[j], A_wq[j], B_wq[j], gamma=1.0, rc=rc_nf)
            D_nf[i, j] = -v_nf[2]

    mirror_of = np.empty(n_pan, dtype=int)
    for i in range(n_pan):
        j = np.argmin(np.abs(y_mid + y_mid[i]))  # y_j ~ -y_i
        mirror_of[i] = j

    computation_params={'D_nf':D_nf,'D_tr':D_tr,'mirror_of':mirror_of, 'c_mid':c_mid, 'y_mid':y_mid,
                      'cbar':cbar,'x_c4_mid':x_c4_mid, 'x_ref':x_ref, 'z_ref':z_ref, 'dy':dy, 'S':S,
                       'n_pan':n_pan, 'tw_mid':tw_mid, 'span': max(y_half)*2.0, 'airfoil_CST':airfoil_CST}

    return computation_params

def run_llt(airfoil_CST, aoa_range, vel_range, airflow, computation_params,
            n_iter=30, beta=0.40, enforce_symmetry=True):

    rows = []
    mach = 0.0
    # geometry/flow 
    dy   = computation_params['dy']          # (n_pan,)
    y    = computation_params['y_mid']       # (n_pan,)
    c    = computation_params['c_mid']       # (n_pan,)
    tw   = computation_params['tw_mid']      # (n_pan,)
    S    = computation_params['S']           # scalar
    cbar = computation_params['cbar']        # scalar
    x_c4 = computation_params['x_c4_mid']    # (n_pan,)
    xref = computation_params['x_ref']       # (n_pan,) or scalar
    span = computation_params['span']        # scalar

    # Influence matrices
    D_nf = computation_params['D_nf']        # (n_pan, n_pan)
    D_tr = computation_params['D_tr']        # (n_pan, n_pan)

    rho   = airflow['rho']
    mu    = airflow['mu']                # 0.5 * rho * V_inf**2


    for aoa_deg in aoa_range:
        for vel in vel_range:
            V_inf = vel
            q_inf=0.5 * rho * V_inf**2
            Re_panels = rho * V_inf * c / mu         # (n_pan,)
            alpha_geo  = aoa_deg + tw                # (n_pan,)

            # --- initial vectorized lookup ---
            aero0 = airfoil_CST.get_aero_from_neuralfoil(alpha=alpha_geo, Re=Re_panels, mach=mach)
            cl = aero0['CL']                         # (n_pan,)
            cd = aero0['CD']                         # (n_pan,)
            Gamma = 0.5 * V_inf * c * cl             # (n_pan,)

            # --- AD-safe Picard iteration: fixed count, no early break ---
            # (You can still compute a residual for monitoring; just don't branch on it.)
            if enforce_symmetry:
                mirror_of = computation_params['mirror_of']  # integer index array

            for _ in range(n_iter):
                w_nf = D_nf @ Gamma                                   # (n_pan,)
                alpha_eff_iter = alpha_geo - np.degrees(np.arctan2(w_nf, V_inf))
                aer = airfoil_CST.get_aero_from_neuralfoil(alpha=alpha_eff_iter, Re=Re_panels, mach=mach)
                cl_star = aer['CL']                                   # (n_pan,)
                Gamma_star = 0.5 * V_inf * c * cl_star
                Gamma_new  = (1.0 - beta) * Gamma + beta * Gamma_star
                if enforce_symmetry:
                    Gamma_new = 0.5 * (Gamma_new + Gamma_new[mirror_of])
                Gamma = Gamma_new

            # --- final fields ---
            w_nf = D_nf @ Gamma
            w_tr = D_tr @ Gamma

            alpha_eff = alpha_geo - np.degrees(np.arctan2(w_tr, V_inf))

            # final section aerodynamics
            aer_final = airfoil_CST.get_aero_from_neuralfoil(alpha=alpha_eff, Re=Re_panels, mach=mach)
            cl = aer_final['CL'] 
            cd = aer_final['CD']
            cm = aer_final['CM']

            # per-unit-span loads
            Lp        = q_inf * c * cl
            Dp_prime  = q_inf * c * cd
            Di_prime  = rho * Gamma * w_tr
            D_total   = Dp_prime + Di_prime

            # totals (reporting)
            L  = rho * V_inf * np.sum(Gamma * dy)     # CL from Γ-integral
            Dp = np.sum(Dp_prime * dy)
            Di = np.sum(Di_prime * dy)

            # Use np.where to avoid value-based branching for AD
            denom_S   = q_inf * np.maximum(S, 1e-30)
            CL  = L  / denom_S
            CDp = Dp / denom_S
            CDi = Di / denom_S
            CD  = CDp + CDi

            # pitching moment about y (nose-up positive): section Cm@c/4 + r×F to ref c/4
            Mprime_c4 = q_inf * (c**2) * cm
            dx = x_c4 - xref
            MxF_y = -(dx * Lp)
            M_pitch = np.sum((Mprime_c4 + MxF_y) * dy)

            denom_cbar = q_inf * np.maximum(S * cbar, 1e-30)
            CM_pitch = M_pitch / denom_cbar

            # roll (x) and yaw (z)
            M_roll = np.sum(y * Lp * dy)               # My = ∫ y L' dy
            M_yaw  = np.sum(y * D_total * dy)          # Mz = ∫ y D' dy

            denom_span = q_inf * np.maximum(S * span, 1e-30)
            CMx = M_roll / denom_span
            CMz = M_yaw  / denom_span

            rows.append({
                    "V_inf": vel, "AoA": aoa_deg,
                    "CL": CL, "CD": CD, "CDi": CDi, "CDp": CDp, 
                    "M_pitch": M_pitch, "M_roll": M_roll, "M_yaw": M_yaw,
                    "CM_pitch":CM_pitch,"CM_roll": CMx, "CM_yaw": CMz  
                })
    res_df = pd.DataFrame(rows)
    return res_df

def run_llt_cuNF_grid(airfoil_cu,
                    aoa_deg_grid,
                    V_inf_grid,
                    airflow,
                    computation_params,
                    n_iter=30,
                    beta=0.40,
                    tol=1e-4,
                    enforce_symmetry=True,
                    device="cuda",
                    model_size="xlarge"):
    """
    Vectorized Weissinger-L + NeuralFoil solver on an AoA-V_inf grid.

    Parameters
    ----------
    aoa_deg_grid : array_like, shape (N_aoa, N_v)
        Geometric AoA in degrees.
    V_inf_grid : array_like, shape (N_aoa, N_v)
        Freestream speeds in m/s.
    airflow : dict
        Needs at least 'rho' and 'mu'.
    computation_params : dict

    Same as solve_iterative_NeuralFoil_grid, but using cuNeuralFoil
    (PyTorch/CUDA) instead of airfoil_CST.get_aero_from_neuralfoil.

    Extra parameters
    ----------------
    device : {"cuda", "cpu"}
        Torch device for cuNeuralFoil.
    model_size : str
        NeuralFoil model size, e.g. "xxsmall", "small", "medium",
        "large", "xlarge", "xxxlarge".
    """

    mach = 0.0

    def _resolve_torch_device(device: str) -> str:
        """
        - "cuda" -> "cuda"
        - "cpu"  -> "cpu"
        - "mps"  -> "mps" (Apple Silicon, Metal backend)
        """
        if device is None:
            return "cpu"

        d = device.lower()
        if d == "cuda":
            return "cuda"
        if d == "cpu":
            return "cpu"
        if d in ["mps", "metal"]:
            return "mps"

        raise ValueError(f"Unknown device string '{device}'. Use 'cuda', 'cpu', or 'mps'.")
    
    device = _resolve_torch_device(device)

    def eval_nf(alpha_3d, Re_3d):
            """
            alpha_3d, Re_3d: shape (N_aoa, N_v, n_pan)
            Returns a dict of arrays reshaped back to alpha_3d.shape.

            Uses a SINGLE batched cuNeuralFoil call:
            - B = N_aoa * N_v * n_pan identical airfoils
            - alpha_batch, Re_batch of shape (B,)
            """
            # Flatten AoA/Re grids: (N_aoa, N_v, n_pan) -> (B,)
            alpha_flat = np.ravel(alpha_3d)
            Re_flat    = np.ravel(Re_3d)
            B = alpha_flat.size

            # Base Kulfan parameters for a single airfoil (already on device)
            # airfoil_cu is a cuKulfanAirfoil
            upper_base = airfoil_cu.upper_weights_cuda.to(device)
            lower_base = airfoil_cu.lower_weights_cuda.to(device)
            LE_base    = airfoil_cu.leading_edge_weight_cuda.to(device)
            TE_base    = airfoil_cu.TE_thickness_cuda.to(device)

            # Geometry batch: repeat the same airfoil B times
            upper_batch = upper_base.unsqueeze(0).repeat(B, 1)  # (B, n_upper)
            lower_batch = lower_base.unsqueeze(0).repeat(B, 1)  # (B, n_lower)
            LE_batch    = LE_base.repeat(B)                     # (B,)
            TE_batch    = TE_base.repeat(B)                     # (B,)

            kulfan_batch = {
                "upper_weights_cuda": upper_batch,
                "lower_weights_cuda": lower_batch,
                "leading_edge_weight_cuda": LE_batch,
                "TE_thickness_cuda": TE_batch,
            }

            # Flow parameter batches: (B,)
            alpha_batch = torch.as_tensor(
                alpha_flat,
                dtype=torch.float32,
                device=device,
            )
            Re_batch = torch.as_tensor(
                Re_flat,
                dtype=torch.float32,
                device=device,
            )

            # Single batched cuNeuralFoil call
            with torch.no_grad():  # for now, no gradients through cuNeuralFoil
                aero_t = get_aero_from_kulfan_parameters_cuda(
                    kulfan_batch,
                    alpha_batch,
                    Re_batch,
                    device=device,
                    model_size=model_size,
                )

            # Convert outputs back to NumPy and reshape to original 3D grid shape
            aero_grid = {}
            for k, v in aero_t.items():
                # Each v is shape (B,)
                if isinstance(v, torch.Tensor):
                    v_np = v.detach().cpu().numpy()
                else:
                    v_np = np.asarray(v)
                aero_grid[k] = v_np.reshape(alpha_3d.shape)

            return aero_grid

    
    # --- geometry / wing parameters ---
    dy   = computation_params["dy"]         # (n_pan,)
    y    = computation_params["y_mid"]      # (n_pan,)
    c    = computation_params["c_mid"]      # (n_pan,)
    tw   = computation_params["tw_mid"]     # (n_pan,)
    S    = computation_params["S"]          # scalar
    cbar = computation_params["cbar"]       # scalar
    x_c4 = computation_params["x_c4_mid"]   # (n_pan,)
    xref = computation_params["x_ref"]      # scalar
    span = computation_params["span"]       # scalar

    D_nf     = computation_params["D_nf"]       # (n_pan, n_pan)
    D_tr     = computation_params["D_tr"]       # (n_pan, n_pan)
    mirror_of = computation_params["mirror_of"] # (n_pan,)

    # --- flow properties ---
    rho = airflow["rho"]
    mu  = airflow["mu"]

    # ASB-numpy arrays
    aoa_deg_grid = np.asarray(aoa_deg_grid)
    V_inf_grid   = np.asarray(V_inf_grid)

    grid_shape = aoa_deg_grid.shape        # (N_aoa, N_v)
    n_pan      = c.shape[0]

    # Expand panel-wise quantities to broadcast with grid
    tw_pan    = tw[None, None, :]          # (1,1,n_pan)
    c_pan     = c[None, None, :]
    dy_pan    = dy[None, None, :]
    y_pan     = y[None, None, :]
    x_c4_pan  = x_c4[None, None, :]

    # Add spanwise axis
    aoa_3d   = aoa_deg_grid[..., None]     # (N_aoa, N_v, 1)
    V_inf_3d = V_inf_grid[..., None]       # (N_aoa, N_v, 1)

    # Geometric alpha and Reynolds per panel
    alpha_geo = aoa_3d + tw_pan            # (N_aoa, N_v, n_pan)
    Re_panels = rho * V_inf_3d * c_pan / mu

    # Initial NF lookup

    aero0 = eval_nf(alpha_geo, Re_panels)
    cl    = aero0["CL"]    # shape (N_aoa, N_v, n_pan)

    # Initial guess for Gamma
    Gamma = 0.5 * V_inf_3d * c_pan * cl    # (N_aoa, N_v, n_pan)

    # Helper: apply influence matrices to last axis
    def apply_D(mat, gamma):
        # mat: (n_pan, n_pan), gamma: (*grid, n_pan) -> (*grid, n_pan)
        return np.tensordot(gamma, mat.T, axes=([-1], [0]))

    # Picard iteration (near-field for stability)

    for _ in range(n_iter):
        w_nf = apply_D(D_nf, Gamma)  # (N_aoa, N_v, n_pan)

        alpha_eff_iter = alpha_geo - np.degrees(np.arctan2(w_nf, V_inf_3d))

        aer = eval_nf(alpha_eff_iter, Re_panels)
        cl_star = aer["CL"]

        Gamma_star = 0.5 * V_inf_3d * c_pan * cl_star

        Gamma_new = (1.0 - beta) * Gamma + beta * Gamma_star

        if enforce_symmetry:
            j = mirror_of  # (n_pan,)
            # Mirror across span by indexing last axis
            Gamma_new = 0.5 * (Gamma_new + Gamma_new[..., j])

        # Global convergence check (optional)
        diff = np.max(np.abs(Gamma_new - Gamma))
        if diff < tol * max(1.0, float(np.max(np.abs(Gamma)))):
            Gamma = Gamma_new
            #print(f"Converged after {_+1} iterations with diff={diff:.6e}")
            break
            
        Gamma = Gamma_new

    # Final downwash
    w_nf = apply_D(D_nf, Gamma)
    w_tr = apply_D(D_tr, Gamma)

    # Final effective alpha from Trefftz
    alpha_eff = alpha_geo - np.degrees(np.arctan2(w_tr, V_inf_3d))

    aer_final = eval_nf(alpha_eff, Re_panels)

    cl = aer_final["CL"]
    cd = aer_final["CD"]
    cm = aer_final["CM"]

    # Confidence: min over span, keep % convention
    conf = aer_final.get(
        "analysis_confidence_percent",
        aer_final.get("analysis_confidence", None),
    )
    if conf is None:
        CONF_MIN = np.full(grid_shape, 100.0)
    else:
        max_conf = np.max(conf)
        scale = np.where(max_conf <= 1.0, 100.0, 1.0)
        conf_pct = scale * conf
        CONF_MIN = np.min(conf_pct, axis=-1)   # min over panels

    # Per-unit-span loads
    q_inf_grid = 0.5 * rho * V_inf_grid**2
    q_inf_3d   = q_inf_grid[..., None]

    Lp       = q_inf_3d * c_pan * cl
    Dp_prime = q_inf_3d * c_pan * cd
    Di_prime = rho * Gamma * w_tr
    D_total  = Dp_prime + Di_prime

    # Integrals over span (last axis)
    L  = rho * V_inf_grid * np.sum(Gamma * dy_pan, axis=-1)
    Dp = np.sum(Dp_prime * dy_pan, axis=-1)
    Di = np.sum(Di_prime * dy_pan, axis=-1)

    CL  = L  / (q_inf_grid * S)
    CDp = Dp / (q_inf_grid * S)
    CDi = Di / (q_inf_grid * S)
    CD  = CDp + CDi

    # Pitching moment about ref c/4
    Mprime_c4 = q_inf_3d * (c_pan ** 2) * cm
    dx = x_c4_pan - xref
    MxF_y = -(dx * Lp)
    M_pitch = np.sum((Mprime_c4 + MxF_y) * dy_pan, axis=-1)
    CM_total = M_pitch / (q_inf_grid * S * cbar)

    # Roll (x) and yaw (z)
    M_roll = np.sum(y_pan * Lp * dy_pan, axis=-1)
    M_yaw  = np.sum(y_pan * D_total * dy_pan, axis=-1)

    CMx = M_roll / (q_inf_grid * S * span)
    CMz = M_yaw  / (q_inf_grid * S * span)

    res_df = pd.DataFrame({
    "V_inf":   V_inf_grid.ravel(),
    "AoA":     aoa_deg_grid.ravel(),
    "CL":      CL.ravel(),
    "CD":      CD.ravel(),
    "CDi":     CDi.ravel(),
    "CDp":     CDp.ravel(),
    "M_pitch": M_pitch.ravel(),
    "M_roll":  M_roll.ravel(),
    "M_yaw":   M_yaw.ravel(),
    "CM_pitch":      CM_total.ravel(),
    "CM_roll":     CMx.ravel(),
    "CM_yaw":     CMz.ravel(),
    })

    return res_df