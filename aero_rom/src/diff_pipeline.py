# diff_pipeline.py
#
# Combined pipeline: generate CSV coefficient surfaces (wing & elevator)
# and save a lightweight checkpoint with metadata so we can rebuild
# differentiable 3D LLT blocks later.
#
# The actual blocks are rebuilt on demand in src/load_blocks.py.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

from src.config import load_config
from src.io_utils import save_surface_csv
from src.llt import run_llt_cuNF_grid
from src.llt import LLT_computational_params as compute_llt_params
from cuneuralfoil.cu_kulfan_airfoil import cuKulfanAirfoil
import torch

# ------------------------------------------------------------------
# 3D geometry metadata (centroids)
# ------------------------------------------------------------------
# We store chordwise centroids for wing/elevator in the checkpoint so
# glider_optimization can override the colleague's heuristic COM model
# when 3D is enabled. This is intentionally cheap and runs once per
# plane configuration (i.e. when you regenerate 3d_blocks.pt).

def _polygon_area_centroid_x(pts: np.ndarray) -> tuple[float, float]:
    """Return (area, x_centroid) of a closed polygon (x,z)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, 0.0
    # ensure closed
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])
    x = pts[:, 0]
    z = pts[:, 1]
    cross = x[:-1] * z[1:] - x[1:] * z[:-1]
    A = 0.5 * float(np.sum(cross))
    if abs(A) < 1e-18:
        return 0.0, float(np.mean(x[:-1]))
    Cx = float(np.sum((x[:-1] + x[1:]) * cross) / (6.0 * A))
    return abs(A), Cx


def _normalized_airfoil_x_centroid(airfoil_name: str) -> float:
    """Chordwise centroid x in normalized chord coordinates (0..1)."""
    # Import here to keep aero_rom optional where possible.
    import aerosandbox as asb
    from src.geometry import normalize_airfoil_name

    af = asb.Airfoil(normalize_airfoil_name(airfoil_name))
    # AeroSandbox: TE->upper->LE->lower->TE, already in chord fraction.
    pts = np.asarray(af.coordinates, dtype=float)
    # Use (x, y) as (x, z) here.
    area, cx = _polygon_area_centroid_x(pts)
    # If centroid is slightly outside [0,1] due to numerical noise, clamp.
    return float(np.clip(cx, 0.0, 1.0))


def _spanwise_centroid_x(y_half, c_half, xle_half, airfoil_name: str) -> float:
    """Approximate 3D chordwise centroid x using volume proxy integration."""
    from src.geometry import mirror_full

    y, c, xle, _ = mirror_full(
        np.asarray(y_half, float),
        np.asarray(c_half, float),
        np.asarray(xle_half, float),
        np.zeros_like(np.asarray(y_half, float)),
    )
    # Use mid-segment integration.
    yA, yB = y[:-1], y[1:]
    cA, cB = c[:-1], c[1:]
    xA, xB = xle[:-1], xle[1:]
    dy = (yB - yA)
    y_mid = 0.5 * (yA + yB)
    c_mid = 0.5 * (cA + cB)
    xle_mid = 0.5 * (xA + xB)

    xbar_norm = _normalized_airfoil_x_centroid(airfoil_name)

    # Volume proxy weight: section area scales with c^2; thickness distribution constant.
    w = (c_mid ** 2) * np.abs(dy)
    x_mid = xle_mid + c_mid * xbar_norm
    denom = float(np.sum(w))
    if denom < 1e-18:
        return float(np.mean(x_mid))
    return float(np.sum(x_mid * w) / denom)

def run_pipeline(
    config_path: str,
    export_surfaces: bool = True,
    export_ckpt: bool = True
) -> Dict[str, Any]:
    """
    Pipeline:
      1) Run non-differentiable LLT + cuNeuralFoil on an (alpha, V) grid
         to generate full 3D coefficient surfaces and save them as CSVs.
      2) Save a tiny checkpoint with metadata (config path, flow ranges, etc.)
         so that differentiable 3D LLT blocks can be rebuilt later.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.
    export_surfaces : bool, default=True
        Whether to export CSV coefficient surfaces.
    export_ckpt : bool, default=True
        Whether to export the checkpoint file (.pt).

    Returns
    -------
    dict with:
      - "alpha_bounds", "alpha_steps"
      - "vel_bounds", "vel_steps"
      - "csv_dir"    : directory with CSV surfaces (if exported)
      - "models_dir" : directory where checkpoint is stored (if exported)
      - "models_path": path to the metadata checkpoint (.pt, if exported)
    """
    cfg = load_config(config_path)
    outdir = Path(cfg.get("output_dir", "artifacts"))
    raw_dir = outdir / "raw_surfaces"
    models_dir = outdir / "models"
    raw_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Flow properties
    # ------------------------------------------------------------------
    R = float(cfg["flow"]["R"])
    T = float(cfg["flow"]["T"])
    p = float(cfg["flow"]["p"])
    rho = float(cfg["flow"]["rho"])
    nu = float(cfg["flow"]["nu"])
    mu = rho * nu

    airflow = {
        "R": R,
        "T": T,
        "p": p,
        "rho": rho,
        "nu": nu,
        "mu": mu,
    }

    # Alpha / velocity ranges
    alphas = np.arange(
        cfg["flow"]["alpha_range"][0],
        cfg["flow"]["alpha_range"][1] + cfg["flow"]["alpha_step"],
        cfg["flow"]["alpha_step"],
    )
    vels = np.arange(
        cfg["flow"]["vel_range"][0],
        cfg["flow"]["vel_range"][1] + cfg["flow"]["vel_step"],
        cfg["flow"]["vel_step"],
    )

    AoA_grid, V_inf_grid = np.meshgrid(alphas, vels, indexing="ij")

    device = cfg.get("device", "cuda")
    model_size = cfg.get("model_size", "xlarge")
    n_iter = cfg.get("n_iter", 15)
    beta = cfg.get("beta", 0.40)
    tol = cfg.get("tol", 1e-6)
    enforce_symmetry = cfg.get("enforce_symmetry", True)

    # ============================================================
    # 1) WING: geometry + CSV surfaces via run_llt_cuNF_grid
    # ============================================================
    comp_wing = compute_llt_params(
        cfg["wing_geometry"]["y_half"],
        cfg["wing_geometry"]["c_half"],
        cfg["wing_geometry"]["xle_half"],
        cfg["wing_geometry"]["twist_half"],
        cfg["wing_geometry"]["airfoil"],
    )

  

    # we don't need gradients here, this is just for the surfaces
    airfoil_wing = cuKulfanAirfoil(
        comp_wing["airfoil_CST"],
        device=device,
        requires_grad=False,
    )

    if export_surfaces:
        df_wing = run_llt_cuNF_grid(
            airfoil_wing,
            AoA_grid,
            V_inf_grid,
            airflow,
            comp_wing,
            n_iter=n_iter,
            beta=beta,
            tol=tol,
            enforce_symmetry=enforce_symmetry,
            device=device,
            model_size=model_size,
        )

        alpha_vals_wing = np.sort(df_wing["AoA"].unique())
        vel_vals_wing = np.sort(df_wing["V_inf"].unique())
        CL_grid_wing = df_wing.pivot(index="AoA", columns="V_inf", values="CL").values
        CD_grid_wing = df_wing.pivot(index="AoA", columns="V_inf", values="CD").values
        CM_grid_wing = df_wing.pivot(index="AoA", columns="V_inf", values="CM_pitch").values

        save_surface_csv(raw_dir / "cl_wing.csv", alpha_vals_wing, vel_vals_wing, CL_grid_wing, "CL")
        save_surface_csv(raw_dir / "cd_wing.csv", alpha_vals_wing, vel_vals_wing, CD_grid_wing, "CD")
        save_surface_csv(raw_dir / "cm_wing.csv", alpha_vals_wing, vel_vals_wing, CM_grid_wing, "CM")

    # ============================================================
    # 2) ELEVATOR: geometry + CSV surfaces
    # ============================================================
    comp_elev = compute_llt_params(
        cfg["elevator_geometry"]["y_half"],
        cfg["elevator_geometry"]["c_half"],
        cfg["elevator_geometry"]["xle_half"],
        cfg["elevator_geometry"]["twist_half"],
        cfg["elevator_geometry"]["airfoil"],
    )

    airfoil_elev = cuKulfanAirfoil(
        comp_elev["airfoil_CST"],
        device=device,
        requires_grad=False,
    )

    if export_surfaces:
        df_elev = run_llt_cuNF_grid(
            airfoil_elev,
            AoA_grid,
            V_inf_grid,
            airflow,
            comp_elev,
            n_iter=n_iter,
            beta=beta,
            tol=tol,
            enforce_symmetry=enforce_symmetry,
            device=device,
            model_size=model_size,
        )

        alpha_vals_elev = np.sort(df_elev["AoA"].unique())
        vel_vals_elev = np.sort(df_elev["V_inf"].unique())
        CL_grid_elev = df_elev.pivot(index="AoA", columns="V_inf", values="CL").values
        CD_grid_elev = df_elev.pivot(index="AoA", columns="V_inf", values="CD").values
        CM_grid_elev = df_elev.pivot(index="AoA", columns="V_inf", values="CM_pitch").values

        save_surface_csv(raw_dir / "cl_elevator.csv", alpha_vals_elev, vel_vals_elev, CL_grid_elev, "CL")
        save_surface_csv(raw_dir / "cd_elevator.csv", alpha_vals_elev, vel_vals_elev, CD_grid_elev, "CD")
        save_surface_csv(raw_dir / "cm_elevator.csv", alpha_vals_elev, vel_vals_elev, CM_grid_elev, "CM")

    # ============================================================
    # 3) Save a *lightweight* checkpoint with metadata only
    # ============================================================
    models_path = models_dir / "3d_blocks.pt"

    # ------------------------------------------------------------------
    # Chordwise centroids (for dynamics COM override in 3D mode)
    # ------------------------------------------------------------------
    # Computed once per plane config; cheap compared to running surfaces.
    wing_centroid_x = _spanwise_centroid_x(
        cfg["wing_geometry"]["y_half"],
        cfg["wing_geometry"]["c_half"],
        cfg["wing_geometry"]["xle_half"],
        cfg["wing_geometry"]["airfoil"],
    )
    elev_centroid_x = _spanwise_centroid_x(
        cfg["elevator_geometry"]["y_half"],
        cfg["elevator_geometry"]["c_half"],
        cfg["elevator_geometry"]["xle_half"],
        cfg["elevator_geometry"]["airfoil"],
    )

    if export_ckpt:
        torch.save(
            {
                # Just enough info to rebuild the blocks later
                "config_path": config_path,
                "device": device,
                "model_size": model_size,
                "n_iter": n_iter,
                "beta": beta,
                "tol": tol,
                "enforce_symmetry": enforce_symmetry,
                # store flow + geometry config so we don't depend on external files changing
                "flow": cfg["flow"],
                "wing_geometry": cfg["wing_geometry"],
                "elevator_geometry": cfg["elevator_geometry"],
                "centroid": {
                    "wing_x": float(wing_centroid_x),
                    "elevator_x": float(elev_centroid_x),
                },
                # whether we want gradients on these shapes by default
                "wing_requires_grad": True,
                "elevator_requires_grad": cfg.get("elevator_requires_grad", False),
            },
            models_path,
        )

    return {
        "alpha_bounds": (float(alphas.min()), float(alphas.max())),
        "alpha_steps": float(cfg["flow"]["alpha_step"]),
        "vel_bounds": (float(vels.min()), float(vels.max())),
        "vel_steps": float(cfg["flow"]["vel_step"]),
        "csv_dir": str(raw_dir) if export_surfaces else None,
        "models_dir": str(models_dir) if export_ckpt else None,
        "models_path": str(models_path) if export_ckpt else None,
    }
