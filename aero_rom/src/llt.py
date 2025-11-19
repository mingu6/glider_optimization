# lifting-line (3D aggregation)

import numpy as np
import src.geometry as geom
import aerosandbox as asb
import pandas as pd

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
    
    airfoil_CST=asb.Airfoil(geom.normalize_airfoil_name(airfoil_name))
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
    z_ref = -0.002 

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
