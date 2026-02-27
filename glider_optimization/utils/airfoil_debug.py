from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import logging
import torch
import numpy as np


_AIRFOIL_DEBUG_ENABLED = False


def set_airfoil_debug_enabled(enabled: bool) -> None:
    global _AIRFOIL_DEBUG_ENABLED
    _AIRFOIL_DEBUG_ENABLED = bool(enabled)


def is_airfoil_debug_enabled() -> bool:
    return _AIRFOIL_DEBUG_ENABLED


def _tensor_stats(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}: None"

    data = tensor.detach()
    if data.numel() == 0:
        return f"{name}: empty"

    finite = torch.isfinite(data)
    finite_count = int(finite.sum().item())
    total_count = data.numel()
    nan_count = int(torch.isnan(data).sum().item())
    inf_count = int(torch.isinf(data).sum().item())

    if finite_count > 0:
        finite_data = data[finite]
        min_val = float(finite_data.min().item())
        max_val = float(finite_data.max().item())
        mean_val = float(finite_data.mean().item())
        norm_val = float(torch.linalg.norm(finite_data).item())
        return (
            f"{name}: shape={tuple(data.shape)} min={min_val:.6e} max={max_val:.6e} "
            f"mean={mean_val:.6e} norm={norm_val:.6e} finite={finite_count}/{total_count} "
            f"nan={nan_count} inf={inf_count}"
        )

    return (
        f"{name}: shape={tuple(data.shape)} finite=0/{total_count} "
        f"nan={nan_count} inf={inf_count}"
    )


def log_airfoil_debug(
    iteration: int,
    stage: str,
    logger: logging.Logger,
    checkpoint_dir: str,
    tensors: Dict[str, Optional[torch.Tensor]],
) -> None:
    if not _AIRFOIL_DEBUG_ENABLED:
        return

    lines = [f"[airfoil-debug] iter={iteration} stage={stage}"]
    for name, tensor in tensors.items():
        lines.append(_tensor_stats(name, tensor))

    payload = "\n".join(lines)
    logger.info(payload)

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "airfoil_debug.log"
    with out_file.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.write("\n")


def _to_scalar(tensor: Optional[torch.Tensor], default: float = 0.0) -> float:
    if tensor is None:
        return float(default)
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return float(default)
    return float(flat[0].item())


def _to_array(tensor: Optional[torch.Tensor]) -> np.ndarray:
    if tensor is None:
        return np.array([])
    return tensor.detach().cpu().numpy().reshape(-1)


def _kulfan_dict(
    upper_weights: Optional[torch.Tensor],
    lower_weights: Optional[torch.Tensor],
    leading_edge_weight: Optional[torch.Tensor],
    te_thickness: Optional[torch.Tensor],
) -> dict:
    return {
        "TE_thickness": _to_scalar(te_thickness),
        "leading_edge_weight": _to_scalar(leading_edge_weight),
        "lower_weights": _to_array(lower_weights),
        "upper_weights": _to_array(upper_weights),
    }


def log_kulfan_parameters(
    iteration: int,
    stage: str,
    logger: logging.Logger,
    checkpoint_dir: str,
    root_upper: Optional[torch.Tensor],
    root_lower: Optional[torch.Tensor],
    root_le: Optional[torch.Tensor],
    root_te: Optional[torch.Tensor],
    tip_upper: Optional[torch.Tensor],
    tip_lower: Optional[torch.Tensor],
    tip_le: Optional[torch.Tensor],
    tip_te: Optional[torch.Tensor],
) -> None:
    if not _AIRFOIL_DEBUG_ENABLED:
        return

    root = _kulfan_dict(root_upper, root_lower, root_le, root_te)
    tip = _kulfan_dict(tip_upper, tip_lower, tip_le, tip_te)

    def _arr(v: np.ndarray) -> str:
        return np.array2string(v, precision=8, separator=", ", max_line_width=1_000_000)

    lines = [
        f"[airfoil-debug-kulfan] iter={iteration} stage={stage}",
        "root_kulfan_parameters:",
        f"  TE_thickness: {root['TE_thickness']}",
        f"  leading_edge_weight: {root['leading_edge_weight']}",
        f"  upper_weights: {_arr(root['upper_weights'])}",
        f"  lower_weights: {_arr(root['lower_weights'])}",
        "tip_kulfan_parameters:",
        f"  TE_thickness: {tip['TE_thickness']}",
        f"  leading_edge_weight: {tip['leading_edge_weight']}",
        f"  upper_weights: {_arr(tip['upper_weights'])}",
        f"  lower_weights: {_arr(tip['lower_weights'])}",
    ]
    payload = "\n".join(lines)

    logger.info(payload)

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "airfoil_debug.log"
    with out_file.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.write("\n")
