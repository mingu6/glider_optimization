# load/validate JSON config

# import json
# from pathlib import Path
# import numpy as np

# def load_config(path):
#     with open(path, "r") as f:
#         cfg = json.load(f)
    
#     return cfg

import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path: str | Path) -> dict:
    """
    Load either:
      - legacy aero_rom JSON config (old schema), OR
      - glider_optimization YAML config (new schema with 'plane:')

    Returns a dict in the SAME schema that aero_rom expects internally.
    No temporary files are created.
    """
    path = Path(path)

    # 1) Read raw file
    if path.suffix.lower() in [".yaml", ".yml"]:
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML configs. pip install pyyaml")
        with path.open("r") as f:
            raw = yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        with path.open("r") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    # 2) If it's already an aero_rom JSON schema, return as-is
    if isinstance(raw, dict) and "wing_geometry" in raw and "flow" in raw:
        return raw

    # 3) Otherwise interpret it as glider_optimization YAML schema
    if not (isinstance(raw, dict) and "plane" in raw):
        raise ValueError(
            "Config must be either aero_rom JSON (with wing_geometry/flow) "
            "or glider_optimization YAML (with plane: ...)."
        )

    return _yaml_to_aerorom_dict(raw)


def _yaml_to_aerorom_dict(cfg: dict) -> dict:
    plane = cfg["plane"]
    flow = plane.get("flow", {})
    wing = plane.get("wing", None)
    if wing is None:
        raise ValueError("YAML plane.wing is required.")
    elev = plane.get("elevator", None)

    nf = cfg.get("neuralFoilSampling", {}) or {}
    run = cfg.get("run", {}) or {}

    rho = float(flow.get("rho", 1.225))
    mu = float(flow.get("mu", 1.789e-5))
    nu = mu / rho

    out = {
        "flow": {
            "rho": rho,
            "mu": mu,
            "nu": nu,                 # diff_pipeline expects this
            # Keep these as defaults unless your YAML provides them
            "R": float(flow.get("R", 287.05)),
            "T": float(flow.get("T", 288.15)),
            "p": float(flow.get("p", rho * float(flow.get("R", 287.05)) * float(flow.get("T", 288.15)))),

            # surface grids (if export_surfaces=True). You can override in YAML later if desired.
            "alpha_range": [float(nf.get("AoA_min", -10.0)), float(nf.get("AoA_max", 25.0))],
            "alpha_step": float(nf.get("alpha_step", 1.0)),
            # vel_range computed below from Re_range and cbar
        },

        "wing_geometry": {
            "y_half": wing["y_half"],
            "c_half": wing["c_half"],
            "xle_half": wing["xle_half"],
            "twist_half": wing["twist_half"],
            "airfoil": wing.get("airfoil", "naca0012"),
        },

        # LLT solver settings (with good defaults)
        "beta": float(nf.get("llt_beta", 0.40)),
        "tol": float(nf.get("llt_tol", 1e-6)),
        "n_iter": int(nf.get("llt_n_iter", 15)),
        "enforce_symmetry": bool(nf.get("llt_enforce_symmetry", True)),
        "device": str(run.get("device", "cpu")),
        "model_size": str(nf.get("llt_model_size", nf.get("neuralFoil_size", "xxxlarge"))),
    }

    if elev is not None:
        out["elevator_geometry"] = {
            "y_half": elev["y_half"],
            "c_half": elev["c_half"],
            "xle_half": elev["xle_half"],
            "twist_half": elev["twist_half"],
            "airfoil": elev.get("airfoil", wing.get("airfoil", "naca0012")),
        }

    # Compute vel_range from Re_range + cbar (matches your LLT convention)
    from src.llt import LLT_computational_params
    comp = LLT_computational_params(
        out["wing_geometry"]["y_half"],
        out["wing_geometry"]["c_half"],
        out["wing_geometry"]["xle_half"],
        out["wing_geometry"]["twist_half"],
        out["wing_geometry"]["airfoil"],
    )
    cbar = float(comp["cbar"])

    Re_min = float(nf.get("Re_min", 1e4))
    Re_max = float(nf.get("Re_max", 6e5))
    V_min = Re_min * mu / (rho * cbar)
    V_max = Re_max * mu / (rho * cbar)

    out["flow"]["vel_range"] = [float(V_min), float(V_max)]
    out["flow"]["vel_step"] = float(nf.get("vel_step", 1.0))

    return out
