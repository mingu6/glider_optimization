"""
load_blocks.py

Utilities to rebuild differentiable 3D LLT blocks (wing & elevator)
from the lightweight checkpoint produced by diff_pipeline.run_pipeline.
"""

from __future__ import annotations

from typing import Dict, Any

import torch

from src.config import load_config
from src.llt import LLT_computational_params as compute_llt_params
from src.diff_llt import CuNFWeissingerLLT
from src.models import ClModel, CdModel, CmModel
from src.implicit_llt import CuNFWeissingerLLTImplicit
from src.implicit_models import ClModelImplicit, CdModelImplicit, CmModelImplicit
from cuneuralfoil.cu_kulfan_airfoil import cuKulfanAirfoil


def load_part_from_ckpt(
    ckpt_path: str,
    part: str = "wing",
    mode: str = "implicit",
    device: str | None = None
) -> Dict[str, Any]:
    """
    Load a single component (wing or elevator) with specified differentiation mode.
    
    This is the unified loader function used by AeroBlock.from_ckpt().
    
    Parameters
    ----------
    ckpt_path : str
        Path to the .pt checkpoint file.
    part : str, default="wing"
        Component to load: "wing" or "elevator".
    mode : str, default="implicit"
        Differentiation mode: "explicit" or "implicit".
    device : str or None, default=None
        Device to place tensors on. If None, uses device from checkpoint.
    
    Returns
    -------
    dict
        Dictionary with keys: airfoil, llt_model, shape_params,
        cl_block, cd_block, cm_block.
    
    Examples
    --------
    >>> blocks = load_part_from_ckpt("model.pt", part="wing", mode="implicit")
    >>> cl_block = blocks["cl_block"]
    """
    assert part in ["wing", "elevator"], f"part must be 'wing' or 'elevator', got '{part}'"
    assert mode in ["explicit", "implicit"], f"mode must be 'explicit' or 'implicit', got '{mode}'"
    
    # Load full blocks dictionary using the appropriate loader
    if mode == "implicit":
        all_blocks = load_blocks_from_ckpt_implicit(ckpt_path, device=device)
    else:
        all_blocks = load_blocks_from_ckpt(ckpt_path, device=device)
    
    # Return only the requested part
    return all_blocks[part]


def load_blocks_from_ckpt(ckpt_path: str, device: str | None = None) -> Dict[str, Any]:
    """
    Rebuild differentiable 3D LLT blocks (wing & elevator) from a checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to the .pt file created by diff_pipeline.run_pipeline.
    device : {"cuda", "cpu", "mps"} or None
        Device to place the models/parameters on. If None, uses the device
        stored in the checkpoint.

    Returns
    -------
    dict with:
      {
        "wing": {
            "airfoil": airfoil_wing,
            "llt_model": model_wing,
            "shape_params": shape_params_wing,
            "cl_block": cl_block_wing,
            "cd_block": cd_block_wing,
            "cm_block": cm_block_wing,
        },
        "elevator": {
            "airfoil": airfoil_elev,
            "llt_model": model_elev,
            "shape_params": shape_params_elev,
            "cl_block": cl_block_elev,
            "cd_block": cd_block_elev,
            "cm_block": cm_block_elev,
        },
      }
    """
    meta = torch.load(ckpt_path, map_location="cpu")

    cfg_path = meta.get("config_path", None)
    if cfg_path is not None:
        cfg = load_config(cfg_path)
        flow_cfg = cfg["flow"]
        wing_geom = cfg["wing_geometry"]
        elev_geom = cfg["elevator_geometry"]
    else:
        # fall back on stored config snapshot
        flow_cfg = meta["flow"]
        wing_geom = meta["wing_geometry"]
        elev_geom = meta["elevator_geometry"]

    if device is None:
        device = meta.get("device", "cuda")

    model_size = meta.get("model_size", "xlarge")
    n_iter = meta.get("n_iter", 15)
    beta = meta.get("beta", 0.40)
    tol = meta.get("tol", 1e-6)
    enforce_symmetry = meta.get("enforce_symmetry", True)

    # Flow props
    R = float(flow_cfg["R"])
    T = float(flow_cfg["T"])
    p = float(flow_cfg["p"])
    rho = float(flow_cfg["rho"])
    nu = float(flow_cfg["nu"])
    mu = rho * nu

    airflow = {
        "R": R,
        "T": T,
        "p": p,
        "rho": rho,
        "nu": nu,
        "mu": mu,
    }

    # ============================================================
    # WING
    # ============================================================
    comp_wing = compute_llt_params(
        wing_geom["y_half"],
        wing_geom["c_half"],
        wing_geom["xle_half"],
        wing_geom["twist_half"],
        wing_geom["airfoil"],
    )

    airfoil_wing = cuKulfanAirfoil(
        comp_wing["airfoil_CST"],
        device=device,
        requires_grad=meta.get("wing_requires_grad", True),
    )

    model_wing = CuNFWeissingerLLT(
        airfoil_cu=airfoil_wing,
        computation_params=comp_wing,
        airflow=airflow,
        n_iter=n_iter,
        beta=beta,
        tol=tol,
        enforce_symmetry=enforce_symmetry,
        device=device,
        model_size=model_size,
    )

    shape_params_wing = [
        airfoil_wing.upper_weights_cuda,
        airfoil_wing.lower_weights_cuda,
        airfoil_wing.leading_edge_weight_cuda,
        airfoil_wing.TE_thickness_cuda,
    ]

    cl_block_wing = ClModel(model_wing, shape_params_wing)
    cd_block_wing = CdModel(model_wing, shape_params_wing)
    cm_block_wing = CmModel(model_wing, shape_params_wing)

    # ============================================================
    # ELEVATOR
    # ============================================================
    comp_elev = compute_llt_params(
        elev_geom["y_half"],
        elev_geom["c_half"],
        elev_geom["xle_half"],
        elev_geom["twist_half"],
        elev_geom["airfoil"],
    )

    airfoil_elev = cuKulfanAirfoil(
        comp_elev["airfoil_CST"],
        device=device,
        requires_grad=meta.get("elevator_requires_grad", False),
    )

    model_elev = CuNFWeissingerLLT(
        airfoil_cu=airfoil_elev,
        computation_params=comp_elev,
        airflow=airflow,
        n_iter=n_iter,
        beta=beta,
        tol=tol,
        enforce_symmetry=enforce_symmetry,
        device=device,
        model_size=model_size,
    )

    shape_params_elev = [
        airfoil_elev.upper_weights_cuda,
        airfoil_elev.lower_weights_cuda,
        airfoil_elev.leading_edge_weight_cuda,
        airfoil_elev.TE_thickness_cuda,
    ]

    cl_block_elev = ClModel(model_elev, shape_params_elev)
    cd_block_elev = CdModel(model_elev, shape_params_elev)
    cm_block_elev = CmModel(model_elev, shape_params_elev)

    return {
        "wing": {
            "airfoil": airfoil_wing,
            "llt_model": model_wing,
            "shape_params": shape_params_wing,
            "cl_block": cl_block_wing,
            "cd_block": cd_block_wing,
            "cm_block": cm_block_wing,
        },
        "elevator": {
            "airfoil": airfoil_elev,
            "llt_model": model_elev,
            "shape_params": shape_params_elev,
            "cl_block": cl_block_elev,
            "cd_block": cd_block_elev,
            "cm_block": cm_block_elev,
        },
    }


def load_blocks_from_ckpt_implicit(ckpt_path: str, device: str | None = None) -> Dict[str, Any]:
    """
    Rebuild differentiable 3D LLT blocks (wing & elevator) from a checkpoint.
    Uses implicit differentiation for efficient gradient computation.

    Parameters
    ----------
    ckpt_path : str
        Path to the .pt file created by diff_pipeline.run_pipeline.
    device : {"cuda", "cpu", "mps"} or None
        Device to place the models/parameters on. If None, uses the device
        stored in the checkpoint.

    Returns
    -------
    dict with:
      {
        "wing": {
            "airfoil": airfoil_wing,
            "llt_model": model_wing,
            "shape_params": shape_params_wing,
            "cl_block": cl_block_wing,
            "cd_block": cd_block_wing,
            "cm_block": cm_block_wing,
        },
        "elevator": {
            "airfoil": airfoil_elev,
            "llt_model": model_elev,
            "shape_params": shape_params_elev,
            "cl_block": cl_block_elev,
            "cd_block": cd_block_elev,
            "cm_block": cm_block_elev,
        },
      }
    """
    meta = torch.load(ckpt_path, map_location="cpu")

    cfg_path = meta.get("config_path", None)
    if cfg_path is not None:
        cfg = load_config(cfg_path)
        flow_cfg = cfg["flow"]
        wing_geom = cfg["wing_geometry"]
        elev_geom = cfg["elevator_geometry"]
    else:
        # fall back on stored config snapshot
        flow_cfg = meta["flow"]
        wing_geom = meta["wing_geometry"]
        elev_geom = meta["elevator_geometry"]

    if device is None:
        device = meta.get("device", "cuda")

    model_size = meta.get("model_size", "xlarge")
    n_iter = meta.get("n_iter", 15)
    beta = meta.get("beta", 0.40)
    tol = meta.get("tol", 1e-6)
    enforce_symmetry = meta.get("enforce_symmetry", True)

    # Flow props
    R = float(flow_cfg["R"])
    T = float(flow_cfg["T"])
    p = float(flow_cfg["p"])
    rho = float(flow_cfg["rho"])
    nu = float(flow_cfg["nu"])
    mu = rho * nu

    airflow = {
        "R": R,
        "T": T,
        "p": p,
        "rho": rho,
        "nu": nu,
        "mu": mu,
    }

    # ============================================================
    # WING
    # ============================================================
    comp_wing = compute_llt_params(
        wing_geom["y_half"],
        wing_geom["c_half"],
        wing_geom["xle_half"],
        wing_geom["twist_half"],
        wing_geom["airfoil"],
    )

    airfoil_wing = cuKulfanAirfoil(
        comp_wing["airfoil_CST"],
        device=device,
        requires_grad=True,  # Always True for implicit differentiation
    )

    model_wing = CuNFWeissingerLLTImplicit(
        airfoil_cu=airfoil_wing,
        computation_params=comp_wing,
        airflow=airflow,
        n_iter=n_iter,
        beta=beta,
        tol=tol,
        enforce_symmetry=enforce_symmetry,
        device=device,
        model_size=model_size,
    )

    shape_params_wing = [
        airfoil_wing.upper_weights_cuda,
        airfoil_wing.lower_weights_cuda,
        airfoil_wing.leading_edge_weight_cuda,
        airfoil_wing.TE_thickness_cuda,
    ]

    cl_block_wing = ClModelImplicit(model_wing, shape_params_wing)
    cd_block_wing = CdModelImplicit(model_wing, shape_params_wing)
    cm_block_wing = CmModelImplicit(model_wing, shape_params_wing)

    # ============================================================
    # ELEVATOR
    # ============================================================
    comp_elev = compute_llt_params(
        elev_geom["y_half"],
        elev_geom["c_half"],
        elev_geom["xle_half"],
        elev_geom["twist_half"],
        elev_geom["airfoil"],
    )

    airfoil_elev = cuKulfanAirfoil(
        comp_elev["airfoil_CST"],
        device=device,
        requires_grad=True,  # Always True for implicit differentiation
    )

    model_elev = CuNFWeissingerLLTImplicit(
        airfoil_cu=airfoil_elev,
        computation_params=comp_elev,
        airflow=airflow,
        n_iter=n_iter,
        beta=beta,
        tol=tol,
        enforce_symmetry=enforce_symmetry,
        device=device,
        model_size=model_size,
    )

    shape_params_elev = [
        airfoil_elev.upper_weights_cuda,
        airfoil_elev.lower_weights_cuda,
        airfoil_elev.leading_edge_weight_cuda,
        airfoil_elev.TE_thickness_cuda,
    ]

    cl_block_elev = ClModelImplicit(model_elev, shape_params_elev)
    cd_block_elev = CdModelImplicit(model_elev, shape_params_elev)
    cm_block_elev = CmModelImplicit(model_elev, shape_params_elev)

    return {
        "wing": {
            "airfoil": airfoil_wing,
            "llt_model": model_wing,
            "shape_params": shape_params_wing,
            "cl_block": cl_block_wing,
            "cd_block": cd_block_wing,
            "cm_block": cm_block_wing,
        },
        "elevator": {
            "airfoil": airfoil_elev,
            "llt_model": model_elev,
            "shape_params": shape_params_elev,
            "cl_block": cl_block_elev,
            "cd_block": cd_block_elev,
            "cm_block": cm_block_elev,
        },
    }
