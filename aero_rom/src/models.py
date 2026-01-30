# 3D aerodynamic coefficient blocks built on top of the differentiable
# CuNFWeissingerLLT model in diff_llt.py.
#
# Each class exposes:
#   - __call__(alpha_deg, V_inf): forward evaluation of the 3D coefficient
#   - backward(alpha_deg, V_inf): gradients of that coefficient w.r.t. a given
#       list of parameter tensors (typically the Kulfan weights from
#       cuKulfanAirfoil, with requires_grad=True).
#
# This replaces the earlier RGI-based ClModel/CdModel/CmModel. No interpolation
# is used here; everything comes from the full LLT + cuNeuralFoil model.

from __future__ import annotations

from typing import Iterable, List
import torch
from src.diff_llt import CuNFWeissingerLLT


class _Base3DCoeff:
    """Base wrapper around a CuNFWeissingerLLT model for a single coefficient.

    Parameters
    ----------
    llt_model : CuNFWeissingerLLT
        Differentiable 3D LLT model (torch.nn.Module).
    coeff_key : str
        Key in the dict returned by llt_model.forward(...), e.g. "CL", "CD", "CM".
    params : Iterable[torch.Tensor]
        List of tensors to differentiate with respect to (e.g. Kulfan weights
        from cuKulfanAirfoil). All of them should have requires_grad=True.
    """

    def __init__(
        self,
        llt_model: CuNFWeissingerLLT,
        coeff_key: str,
        params: Iterable[torch.Tensor],
    ) -> None:
        self.llt_model = llt_model
        self.coeff_key = coeff_key
        self.params: List[torch.Tensor] = list(params)

    # ---------------------------------------------
    # Forward: value of the coefficient at (α, V)
    # ---------------------------------------------
    def __call__(self, alpha_deg, V_inf):
        """Evaluate the 3D coefficient at a given (alpha_deg, V_inf).

        Returns
        -------
        float
            If scalar inputs are given, returns a Python float.
        torch.Tensor
            If tensor inputs are given, returns a tensor of matching shape.
        """
        out = self.llt_model(alpha_deg, V_inf)[self.coeff_key]
        # If this is a scalar tensor, return a float for convenience
        if isinstance(out, torch.Tensor) and out.numel() == 1:
            return float(out.item())
        return out

    # ---------------------------------------------
    # Backward: gradients wrt shape parameters
    # ---------------------------------------------
    def backward(
        self,
        alpha_deg,
        V_inf,
        *,
        v=None,
        tangent=None,
        return_dict: bool = False
    ):
        """Compute gradients of this coefficient wrt the chosen parameters.

        This uses PyTorch autograd under the hood. The gradients are with
        respect to the tensors given in `params` at construction time.
        
        Supports VJP (vector-Jacobian product) and JVP (Jacobian-vector product)
        for integration into larger differentiable pipelines.

        Parameters
        ----------
        alpha_deg : float or tensor
        V_inf : float or tensor
        v : float or torch.Tensor, optional
            Upstream gradient scalar for VJP computation.
            If provided, returns v * dC/dp instead of dC/dp.
        tangent : list[torch.Tensor], optional
            Tangent vectors for JVP computation.
            Returns dot(dC/dp, tangent) as a scalar.
        return_dict : bool, default=False
            If True, return dict with keys "grads", "vjp", "jvp".
            If False (default), return grads or VJP directly.

        Returns
        -------
        grads : list[torch.Tensor]
            List of gradients dC/dp for each parameter tensor p in `params`.
            If a parameter is not connected to the coefficient, its gradient
            will be returned as None.
        OR dict (if return_dict=True)
            {"grads": [...], "vjp": [...] or None, "jvp": scalar or None}
        """
        # Ensure no stale gradients
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

        # Forward pass; keep the tensor (do not convert to float here)
        val = self.llt_model(alpha_deg, V_inf)[self.coeff_key]
        # If val is a vector, sum to get a scalar objective
        if isinstance(val, torch.Tensor) and val.ndim > 0:
            val = val.sum()

        grads = torch.autograd.grad(
            val,
            self.params,
            retain_graph=False,
            allow_unused=True,
        )
        grads = list(grads)

        # Compute VJP if v is provided
        vjp = None
        if v is not None:
            v_scalar = float(v) if not isinstance(v, torch.Tensor) else v.item()
            vjp = [v_scalar * g if g is not None else None for g in grads]
        
        # Compute JVP if tangent is provided
        jvp = None
        if tangent is not None:
            jvp = 0.0
            for g, t in zip(grads, tangent):
                if g is not None and t is not None:
                    jvp += (g * t).sum().item()
        
        # Return based on format requested
        if return_dict:
            return {"grads": grads, "vjp": vjp, "jvp": jvp}
        else:
            # Default: return VJP if v provided, else raw gradients
            return vjp if vjp is not None else grads


class ClModel(_Base3DCoeff):
    """3D lift coefficient block (CL)."""
    def __init__(self, llt_model: CuNFWeissingerLLT, params: Iterable[torch.Tensor]) -> None:
        super().__init__(llt_model, coeff_key="CL", params=params)


class CdModel(_Base3DCoeff):
    """3D drag coefficient block (CD)."""
    def __init__(self, llt_model: CuNFWeissingerLLT, params: Iterable[torch.Tensor]) -> None:
        super().__init__(llt_model, coeff_key="CD", params=params)


class CmModel(_Base3DCoeff):
    """3D pitching moment coefficient block (CM)."""
    def __init__(self, llt_model: CuNFWeissingerLLT, params: Iterable[torch.Tensor]) -> None:
        super().__init__(llt_model, coeff_key="CM", params=params)
