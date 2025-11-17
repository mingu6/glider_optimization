import numpy as np
import json
import pandas as pd
from scipy.optimize import least_squares

def interpolate_with_anchors(y_known, fields, y_query, *, allow_extrapolation=False, atol=1e-12):
    """
    y_known: 1D array of anchor y-stations (original geometry)
    fields:  dict mapping name -> 1D array of values at y_known
             e.g. {"c": c_half, "xle": xle_half, "twist": twist_half}
    y_query: 1D array of y where you also want values
    allow_extrapolation: if False, error when y_query is outside [min(y_known), max(y_known)]
    atol: tolerance when matching y_known back into the merged grid
    Returns: dict with "y" and each field on the merged, sorted grid
    """
    y_known = np.asarray(y_known, float)
    y_query = np.asarray(y_query, float)

    # sort anchors
    order = np.argsort(y_known)
    yk = y_known[order]

    # range check
    if not allow_extrapolation:
        if y_query.min() < yk.min() - 1e-15 or y_query.max() > yk.max() + 1e-15:
            raise ValueError("Query y contains points outside the known range. Set allow_extrapolation=True to override.")

    # merged, sorted y grid (anchors + targets)
    y_out = np.unique(np.concatenate([yk, y_query]))

    out = {"y": y_out}
    for name, vals in fields.items():
        vk = np.asarray(vals, float)[order]
        v_out = np.interp(y_out, yk, vk)  # linear interpolation
        # Re-impose exact anchor values (avoid any FP drift)
        for yi, vi in zip(yk, vk):
            idx = np.where(np.isclose(y_out, yi, atol=atol))[0]
            if idx.size:
                v_out[idx[0]] = vi
        out[name] = v_out

    return out

def print_for_paste(name, arr, fmt=".6f"):
    """
    Pretty-print as: name = np.array([a, b, c])
    Trailing zeros and dots are trimmed for clean pasting.
    """
    arr = np.asarray(arr)
    s = ", ".join(f"{v:{fmt}}".rstrip("0").rstrip(".") for v in arr)
    print(f"{name} = np.array([{s}])")

def load_comp(csv_path: str, suffix: str) -> pd.DataFrame:
    """
    Read a component aero CSV and return columns:
    AoA, CL_<suffix>, CD_<suffix>, CM_<suffix>
    Accepts CD or (CDi + CDp). Duplicated AoA rows are averaged.
    """
    df = pd.read_csv(csv_path)

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    aoa_col = pick('AoA', 'aoa', 'AOA')
    cl_col  = pick('CL', 'Cl', 'cl')
    cm_col  = pick('CM', 'Cm', 'cm')

    if aoa_col is None or cl_col is None or cm_col is None:
        raise KeyError(f"Expected columns like AoA/CL/CM in {csv_path}; got {list(df.columns)}")

    if 'CD' in df.columns:
        cd_series = df['CD']
    else:
        cdi_col = pick('CDi', 'Cdi', 'cdi')
        cdp_col = pick('CDp', 'Cdp', 'cdp')
        if cdi_col is None or cdp_col is None:
            raise KeyError(f"No CD or CDi+CDp in {csv_path}; got {list(df.columns)}")
        cd_series = df[cdi_col] + df[cdp_col]

    out = pd.DataFrame({
        'AoA': df[aoa_col].astype(float),
        f'CL_{suffix}': df[cl_col].astype(float),
        f'CD_{suffix}': cd_series.astype(float),
        f'CM_{suffix}': df[cm_col].astype(float)
    })

    # If any duplicate AoA rows exist, average them; then sort.
    out = out.groupby('AoA', as_index=False).mean().sort_values('AoA').reset_index(drop=True)
    return out

def mirror_full(y, c, xle, twist):
    y_full   = np.concatenate((-y[::-1], y[1:]))
    c_full   = np.concatenate(( c[::-1],  c[1:]))
    xle_full = np.concatenate((xle[::-1], xle[1:]))
    tw_full  = np.concatenate((twist[::-1], twist[1:]))
    # sort by y
    o = np.argsort(y_full)
    return y_full[o], c_full[o], xle_full[o], tw_full[o]

def mac_from_geometry(y_full, c_full):
    """
    Returns (S, cbar) using exact segment integrals for linear chord between nodes.
    S = ∫ c dy;  ∫ c^2 dy over [yA,yB] = Δy/3 (cA^2 + cA*cB + cB^2).
    """
    yA, yB = y_full[:-1], y_full[1:]
    cA, cB = c_full[:-1], c_full[1:]
    dy = yB - yA
    S    = np.sum(0.5*(cA + cB) * dy)
    I2   = np.sum((dy/3.0) * (cA*cA + cA*cB + cB*cB))   # ∫ c^2 dy
    cbar = I2 / S
    return float(S), float(cbar)

def load_geom(path):
    with open(path, "r") as f: g = json.load(f)
    yh = np.array(g["y_half"], dtype=float)
    ch = np.array(g["c_half"], dtype=float)
    xh = np.array(g["xle_half"], dtype=float)
    th = np.array(g["twist_half"], dtype=float)
    return yh, ch, xh, th

def read_dat(path):
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()
    lines = [ln.strip() for ln in lines if ln.strip() != ""]
    data = []
    for ln in lines:
        try:
            parts = ln.replace(",", " ").split()
            if len(parts) < 2:
                continue
            x = float(parts[0]); z = float(parts[1])
            data.append([x, z])
        except:
            continue
    return np.array(data, dtype=float)

def normalize_and_split(points):
    x = points[:,0]; z = points[:,1]
    idx_le = np.argmin(x); idx_te = np.argmax(x)
    chord = x[idx_te] - x[idx_le]
    xn = (x - x[idx_le]) / chord
    zn = z
    upper = np.stack([xn[:idx_le+1], zn[:idx_le+1]], axis=1)
    lower = np.stack([xn[idx_le:],  zn[idx_le:]],  axis=1)
    upper = upper[np.argsort(upper[:,0])]
    lower = lower[np.argsort(lower[:,0])]
    def dedup(arr):
        _, unique_idx = np.unique(np.round(arr[:,0], 8), return_index=True)
        return arr[np.sort(unique_idx)]
    return dedup(upper), dedup(lower)

def class_function(x, N1=0.5, N2=1.0):
    x = np.clip(x, 1e-9, 1-1e-9)
    return (x**N1) * ((1-x)**N2)

def bernstein_matrix(n, x):
    M = np.zeros((len(x), n+1))
    from math import comb
    for i in range(n+1):
        M[:, i] = comb(n, i) * (x**i) * ((1-x)**(n-i))
    return M

def cst_surface(x, A, delta_te, N1=0.5, N2=1.0, sign=+1):
    C = class_function(x, N1, N2)
    n = len(A) - 1
    B = bernstein_matrix(n, x)
    S = B @ A
    return C * S + sign * 0.5 * x * delta_te

def pack_params(Au, Al, delta_te):
    return np.concatenate([Au, Al, np.array([delta_te])])

def unpack_params(p, n):
    Au = p[:n+1]
    Al = p[n+1:2*(n+1)]
    delta_te = p[-1]
    return Au, Al, delta_te

def fit_cst(upper, lower, n=8, N1=0.5, N2=1.0, w_le=3.0):
    xu, yu = upper[:,0], upper[:,1]
    xl, yl = lower[:,0], lower[:,1]
    Au0 = np.zeros(n+1); Al0 = np.zeros(n+1); delta0 = 0.0
    p0 = pack_params(Au0, Al0, delta0)
    wu = 1.0 + (w_le - 1.0) * np.exp(-((xu - 0.0)/0.15)**2)
    wl = 1.0 + (w_le - 1.0) * np.exp(-((xl - 0.0)/0.15)**2)

    def residuals(p):
        Au, Al, delta_te = unpack_params(p, n)
        yu_fit = cst_surface(xu, Au, delta_te, N1, N2, sign=+1)
        yl_fit = cst_surface(xl, Al, delta_te, N1, N2, sign=-1)
        return np.concatenate([(yu_fit - yu) * wu, (yl_fit - yl) * wl])

    big = 5.0
    lb = np.full_like(p0, -big); ub = np.full_like(p0, big)
    lb[-1] = -0.05; ub[-1] = 0.05
    result = least_squares(residuals, p0, bounds=(lb, ub),
                           xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
    Au, Al, delta_te = unpack_params(result.x, n)
    return dict(Au=Au, Al=Al, delta_te=delta_te, N1=N1, N2=N2, n=n)

def normalize_airfoil_name(name: str) -> str:
    # lowercase, remove spaces and underscores
    return name.lower().replace("_", "").replace(" ", "")

