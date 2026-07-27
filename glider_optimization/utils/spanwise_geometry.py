from typing import Any, Dict

import aerosandbox as asb
import numpy as np
import torch

def build_half_wing_stations_from_cfg(wing_cfg: Dict[str, Any], n_span_stations: int = 7) -> Dict[str, np.ndarray]:
    """Interpolate the root/tip wing geometry given in `plane.wing` config into
    `n_span_stations` evenly-spaced half-wing stations, root to tip."""
    wing_cfg = wing_cfg or {}
    y0, y1 = wing_cfg.get("y_half", [0.0, 0.42])
    c0, c1 = wing_cfg.get("c_half", [0.1875, 0.1125])
    x0, x1 = wing_cfg.get("xle_half", [0.0, 0.0])
    t0, t1 = wing_cfg.get("twist_half", [0.0, 0.0])

    return {
        "y_half": np.linspace(y0, y1, n_span_stations),
        "c_half": np.linspace(c0, c1, n_span_stations),
        "xle_half": np.linspace(x0, x1, n_span_stations),
        "twist_half": np.linspace(t0, t1, n_span_stations),
        "dihedral_deg": float(wing_cfg.get("dihedral", 0.0)),
    }


def mix_root_tip(root: torch.Tensor, tip: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Linear root->tip interpolation indexed by spanwise fraction `eta` (n_pan,).

    Handles both Kulfan weight vectors (root/tip shape (8,), returns (n_pan, 8)) and
    scalars such as leading-edge weight or TE thickness (shape (1,), returns (n_pan,)).
    """
    root, tip = root.reshape(-1), tip.reshape(-1)
    if root.numel() == 1:
        return (1.0 - eta) * root[0] + eta * tip[0]
    return (1.0 - eta)[:, None] * root[None, :] + eta[:, None] * tip[None, :]


def _polygon_centroid_x(x: np.ndarray, z: np.ndarray) -> float:
    """x-coordinate of the centroid of a closed polygon (airfoil cross-section)."""
    if not (np.isclose(x[0], x[-1]) and np.isclose(z[0], z[-1])):
        x, z = np.append(x, x[0]), np.append(z, z[0])
    cross = x[:-1] * z[1:] - x[1:] * z[:-1]
    return float(np.sum((x[:-1] + x[1:]) * cross) / (3.0 * np.sum(cross)))


def _section_centroid_x(kulfan: Dict[str, Any]) -> float:
    """x/c centroid of a Kulfan airfoil cross-section (0 = leading edge, 1 = trailing edge)."""
    af = asb.KulfanAirfoil(
        name="tmp",
        upper_weights=np.asarray(kulfan["upper_weights"], dtype=float),
        lower_weights=np.asarray(kulfan["lower_weights"], dtype=float),
        leading_edge_weight=float(kulfan["leading_edge_weight"]),
        TE_thickness=float(kulfan["TE_thickness"]),
    )
    return _polygon_centroid_x(np.asarray(af.x(), dtype=float), np.asarray(af.y(), dtype=float))


def dynamic_centroid_offset(
    wing_cfg: Dict[str, Any],
    root_kulfan: Dict[str, Any],
    tip_kulfan: Dict[str, Any],
    n_span_stations: int = 7,
) -> float:
    """How far aft of the wing's quarter-chord line the structural (mass) centroid
    sits, chord-weighted over the span, given the current root/tip airfoil shapes.

    This is an offset relative to the quarter-chord line, not an absolute position,
    so it can be added directly to the fixed aerodynamic-center arm to get a
    shape-dependent CoM arm without assuming the wing-local frame (xle=0 at root)
    and the body frame share an origin.
    """
    stations = build_half_wing_stations_from_cfg(wing_cfg, n_span_stations=n_span_stations)
    y, c, xle = stations["y_half"], stations["c_half"], stations["xle_half"]

    eta = np.clip(y / max(y[-1], 1e-12), 0.0, 1.0)
    cx_root, cx_tip = _section_centroid_x(root_kulfan), _section_centroid_x(tip_kulfan)
    x_centroid = xle + c * ((1.0 - eta) * cx_root + eta * cx_tip)
    x_quarter_chord = xle + 0.25 * c

    norm = np.trapezoid(c, y)
    return float(np.trapezoid(c * (x_centroid - x_quarter_chord), y) / norm)
