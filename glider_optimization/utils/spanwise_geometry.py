from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import aerosandbox as asb
import torch


def _endpoints(values: Any, fallback_start: float, fallback_end: float) -> tuple[float, float]:
    if isinstance(values, list) and len(values) >= 2:
        return float(values[0]), float(values[-1])
    return float(fallback_start), float(fallback_end)


def build_half_wing_stations_from_cfg(wing_cfg: Dict[str, Any], n_span_stations: int = 7) -> Dict[str, np.ndarray]:
    """
    Build half-wing stations using the same endpoint + linspace rule used by the 3D LLT block.
    """
    y_src = wing_cfg.get("y_half", [0.0, 0.42]) if isinstance(wing_cfg, dict) else [0.0, 0.42]
    c_src = wing_cfg.get("c_half", [0.1875, 0.1125]) if isinstance(wing_cfg, dict) else [0.1875, 0.1125]
    xle_src = wing_cfg.get("xle_half", [0.0, 0.0]) if isinstance(wing_cfg, dict) else [0.0, 0.0]
    twist_src = wing_cfg.get("twist_half", [0.0, 0.0]) if isinstance(wing_cfg, dict) else [0.0, 0.0]

    y0, y1 = _endpoints(y_src, 0.0, 0.42)
    c0, c1 = _endpoints(c_src, 0.1875, 0.1125)
    x0, x1 = _endpoints(xle_src, 0.0, 0.0)
    t0, t1 = _endpoints(twist_src, 0.0, 0.0)

    dihedral_deg = float(wing_cfg.get("dihedral", 0.0)) if isinstance(wing_cfg, dict) else 0.0

    y_half = np.linspace(y0, y1, int(n_span_stations), dtype=float)
    c_half = np.linspace(c0, c1, int(n_span_stations), dtype=float)
    xle_half = np.linspace(x0, x1, int(n_span_stations), dtype=float)
    twist_half = np.linspace(t0, t1, int(n_span_stations), dtype=float)
    z_half = y_half * np.tan(np.deg2rad(dihedral_deg))

    return {
        "y_half": y_half,
        "c_half": c_half,
        "xle_half": xle_half,
        "twist_half": twist_half,
        "z_half": z_half,
        "dihedral_deg": dihedral_deg,
    }


def mix_root_tip_torch(root: torch.Tensor, tip: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    Linear root->tip interpolation for torch tensors using spanwise coordinate eta in [0,1].
    """
    if root.ndim == 1:
        return (1.0 - eta)[:, None] * root[None, :] + eta[:, None] * tip[None, :]
    return (1.0 - eta) * root + eta * tip


def _polygon_centroid(x: Iterable[float], z: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(x), dtype=float).reshape(-1)
    z = np.asarray(list(z), dtype=float).reshape(-1)
    if x.size < 3:
        return float("nan"), float("nan")

    if not (np.isclose(x[0], x[-1]) and np.isclose(z[0], z[-1])):
        x = np.r_[x, x[0]]
        z = np.r_[z, z[0]]

    cross = x[:-1] * z[1:] - x[1:] * z[:-1]
    area2 = np.sum(cross)
    if np.isclose(area2, 0.0):
        return float("nan"), float("nan")

    cx = np.sum((x[:-1] + x[1:]) * cross) / (3.0 * area2)
    cz = np.sum((z[:-1] + z[1:]) * cross) / (3.0 * area2)
    return float(cx), float(cz)


def _section_centroid_from_kulfan(kulfan: Dict[str, Any]) -> tuple[float, float]:
    af = asb.KulfanAirfoil(
        name="tmp",
        upper_weights=np.asarray(kulfan["upper_weights"], dtype=float),
        lower_weights=np.asarray(kulfan["lower_weights"], dtype=float),
        leading_edge_weight=float(kulfan["leading_edge_weight"]),
        TE_thickness=float(kulfan["TE_thickness"]),
    )
    x = np.asarray(af.x(), dtype=float).reshape(-1)
    z = np.asarray(af.y(), dtype=float).reshape(-1)
    return _polygon_centroid(x, z)


def compute_dynamic_wing_reference_geometry(
    *,
    wing_cfg: Dict[str, Any],
    root_kulfan: Dict[str, Any],
    tip_kulfan: Dict[str, Any],
    n_span_stations: int = 7,
) -> Dict[str, float]:
    """
    Compute spanwise-interpolated wing reference geometry from current root/tip Kulfan sections.
    """
    stations = build_half_wing_stations_from_cfg(wing_cfg, n_span_stations=n_span_stations)
    y_half = stations["y_half"]
    c_half = stations["c_half"]
    xle_half = stations["xle_half"]

    half_span = float(max(y_half[-1], 1e-12))
    eta = np.clip(y_half / half_span, 0.0, 1.0)

    cx_root, cz_root = _section_centroid_from_kulfan(root_kulfan)
    cx_tip, cz_tip = _section_centroid_from_kulfan(tip_kulfan)

    cx_span = (1.0 - eta) * cx_root + eta * cx_tip
    cz_span = (1.0 - eta) * cz_root + eta * cz_tip

    # Convert centroid from x/c to absolute x along the body axis.
    x_centroid_span = xle_half + c_half * cx_span

    den = float(np.trapezoid(c_half, y_half))
    if den <= 0.0:
        l_w_m = float(np.mean(x_centroid_span))
    else:
        l_w_m = float(np.trapezoid(c_half * x_centroid_span, y_half) / den)

    # Chord-weighted mean z-height of the wing centroid (dihedral arm for inertia)
    dihedral_deg = float(wing_cfg.get("dihedral", 0.0)) if isinstance(wing_cfg, dict) else 0.0
    z_half = y_half * np.tan(np.deg2rad(dihedral_deg))
    l_w_z = float(np.trapezoid(c_half * z_half, y_half) / den) if den > 0.0 else 0.0

    S_half = float(np.trapezoid(c_half, y_half))
    span = float(2.0 * y_half[-1]) if y_half[-1] > 0 else 0.0
    S_w = float(2.0 * S_half) if S_half > 0.0 else 0.0
    chord_ref = float(S_w / span) if span > 0.0 else float(np.mean(c_half))

    return {
        "enabled": True,
        "n_span_stations": int(n_span_stations),
        "l_w_m": l_w_m,
        "chord_ref": float(chord_ref),
        "S_w": float(S_w),
        "cx_root": float(cx_root),
        "cz_root": float(cz_root),
        "cx_tip": float(cx_tip),
        "cz_tip": float(cz_tip),
        "cx_wing": float(np.mean(cx_span)),
        "cz_wing": float(np.mean(cz_span)),
        "dihedral_deg": dihedral_deg,
        "l_w_z": l_w_z,
    }
