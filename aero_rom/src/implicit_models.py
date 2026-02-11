"""
implicit_models.py

Coefficient blocks wrapping the implicit LLT model (CuNFWeissingerLLTImplicit).

Same API style as your current blocks:
  - __call__(alpha, V) -> float or tensor
  - backward(alpha, V) -> gradients w.r.t. chosen parameters

This file does NOT replace src/models.py. It's an additional option.
"""

from __future__ import annotations

from typing import Iterable, List
import torch

from src.implicit_llt import CuNFWeissingerLLTImplicit


class _Base3DCoeffImplicit:
    def __init__(
        self,
        llt_model: CuNFWeissingerLLTImplicit,
        coeff_key: str,
        params: Iterable[torch.Tensor],
    ) -> None:
        self.llt_model = llt_model
        self.coeff_key = coeff_key
        self.params: List[torch.Tensor] = list(params)

    def __call__(self, alpha_deg, V_inf):
        out = self.llt_model(alpha_deg, V_inf)[self.coeff_key]
        if isinstance(out, torch.Tensor) and out.numel() == 1:
            return float(out.item())
        return out

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
        
        Uses implicit differentiation (IFT) via CuNFWeissingerLLTImplicit.
        Supports VJP and JVP for integration into larger differentiable pipelines.
        
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
        list[torch.Tensor] or dict
            Gradients w.r.t. parameters, or dict if return_dict=True.
        """
        # zero old grads
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

        val = self.llt_model(alpha_deg, V_inf)[self.coeff_key]
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


class ClModelImplicit(_Base3DCoeffImplicit):
    def __init__(self, llt_model: CuNFWeissingerLLTImplicit, params: Iterable[torch.Tensor]) -> None:
        super().__init__(llt_model, coeff_key="CL", params=params)


class CdModelImplicit(_Base3DCoeffImplicit):
    def __init__(self, llt_model: CuNFWeissingerLLTImplicit, params: Iterable[torch.Tensor]) -> None:
        super().__init__(llt_model, coeff_key="CD", params=params)


class CmModelImplicit(_Base3DCoeffImplicit):
    def __init__(self, llt_model: CuNFWeissingerLLTImplicit, params: Iterable[torch.Tensor]) -> None:
        super().__init__(llt_model, coeff_key="CM", params=params)
