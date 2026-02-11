"""
aero_block.py

Unified aerodynamic block interface for optimal control integration.

Provides a single entry point to differentiable 3D aerodynamics with:
- Mode selection: "explicit" (unrolled) or "implicit" (IFT/adjoint)
- Part selection: "wing" or "elevator"
- Forward evaluation: CL, CD, CM at (alpha, V)
- Backward evaluation: VJP for gradient propagation in trajectory optimization

Example Usage:
--------------
    from src.aero_block import AeroBlock
    
    # Load block with implicit differentiation
    aero = AeroBlock.from_ckpt(
        "artifacts/models/3d_blocks.pt",
        part="wing",
        mode="implicit",
        device="cuda"
    )
    
    # Forward pass
    coeffs = aero.forward(alpha=5.0, V=18.0)
    print(f"CL={coeffs['CL']}, CD={coeffs['CD']}, CM={coeffs['CM']}")
    
    # Backward pass with upstream gradients (VJP)
    grads = aero.backward_combined(
        alpha=5.0,
        V=18.0,
        v_CL=0.8,
        v_CD=-0.3,
        v_CM=0.1
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import torch


class AeroBlock:
    """
    Unified aerodynamic block for trajectory optimization.
    
    This class wraps the explicit or implicit LLT solver and provides
    a clean interface for computing aerodynamic coefficients and their
    gradients with respect to shape parameters.
    
    Attributes
    ----------
    part : str
        Component name ("wing" or "elevator").
    mode : str
        Differentiation mode ("explicit" or "implicit").
    blocks : dict
        Dictionary containing cl_block, cd_block, cm_block, and shape_params.
    cl_block, cd_block, cm_block
        Individual coefficient blocks (ClModel, CdModel, CmModel).
    shape_params : list[torch.Tensor]
        Shape parameters (Kulfan weights) with requires_grad=True.
    """
    
    def __init__(self, part_blocks: Dict, part: str, mode: str):
        """
        Initialize AeroBlock from loaded part blocks.
        
        Parameters
        ----------
        part_blocks : dict
            Dictionary with keys: cl_block, cd_block, cm_block, shape_params, etc.
        part : str
            Component name ("wing" or "elevator").
        mode : str
            Differentiation mode ("explicit" or "implicit").
        """
        self.part = part
        self.mode = mode
        self.blocks = part_blocks
        
        self.cl_block = part_blocks["cl_block"]
        self.cd_block = part_blocks["cd_block"]
        self.cm_block = part_blocks["cm_block"]
        self.shape_params = part_blocks["shape_params"]
        self.llt_model = part_blocks["llt_model"]
    
    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str,
        part: str = "wing",
        mode: str = "implicit",
        device: Optional[str] = None
    ) -> AeroBlock:
        """
        Load AeroBlock from checkpoint with specified mode and part.
        
        Parameters
        ----------
        ckpt_path : str
            Path to the .pt checkpoint file.
        part : str, default="wing"
            Component to load ("wing" or "elevator").
        mode : str, default="implicit"
            Differentiation mode ("explicit" or "implicit").
        device : str or None, default=None
            Device to place tensors on ("cuda", "cpu", "mps").
            If None, uses device stored in checkpoint.
        
        Returns
        -------
        AeroBlock
            Initialized aerodynamic block ready for forward/backward evaluation.
        
        Examples
        --------
        >>> aero = AeroBlock.from_ckpt("model.pt", part="wing", mode="implicit")
        >>> CL = aero.cl(alpha=5.0, V=18.0)
        """
        from .load_blocks import load_part_from_ckpt
        
        part_blocks = load_part_from_ckpt(
            ckpt_path,
            part=part,
            mode=mode,
            device=device
        )
        return cls(part_blocks, part=part, mode=mode)
    
    # =========================================================================
    # Forward evaluation methods
    # =========================================================================
    
    def cl(self, alpha: float, V: float) -> float:
        """
        Evaluate lift coefficient at (alpha, V).
        
        Parameters
        ----------
        alpha : float
            Angle of attack in degrees.
        V : float
            Velocity in m/s.
        
        Returns
        -------
        float
            3D lift coefficient CL.
        """
        return self.cl_block(alpha, V)
    
    def cd(self, alpha: float, V: float) -> float:
        """
        Evaluate drag coefficient at (alpha, V).
        
        Parameters
        ----------
        alpha : float
            Angle of attack in degrees.
        V : float
            Velocity in m/s.
        
        Returns
        -------
        float
            3D drag coefficient CD.
        """
        return self.cd_block(alpha, V)
    
    def cm(self, alpha: float, V: float) -> float:
        """
        Evaluate pitching moment coefficient at (alpha, V).
        
        Parameters
        ----------
        alpha : float
            Angle of attack in degrees.
        V : float
            Velocity in m/s.
        
        Returns
        -------
        float
            3D pitching moment coefficient CM.
        """
        return self.cm_block(alpha, V)
    
    def forward(self, alpha: float, V: float) -> Dict[str, float]:
        """
        Evaluate all aerodynamic coefficients.
        
        Parameters
        ----------
        alpha : float
            Angle of attack in degrees.
        V : float
            Velocity in m/s.
        
        Returns
        -------
        dict
            Dictionary with keys "CL", "CD", "CM" and float values.
        
        Examples
        --------
        >>> coeffs = aero.forward(alpha=5.0, V=18.0)
        >>> print(f"CL={coeffs['CL']:.4f}")
        """
        return {
            "CL": self.cl(alpha, V),
            "CD": self.cd(alpha, V),
            "CM": self.cm(alpha, V)
        }
    
    # =========================================================================
    # Backward evaluation methods (gradient computation)
    # =========================================================================
    
    def backward(
        self,
        coeff: str,
        alpha: float,
        V: float,
        v: Optional[Union[float, torch.Tensor]] = None,
        tangent: Optional[List[torch.Tensor]] = None,
        return_dict: bool = False
    ) -> Union[List[torch.Tensor], Dict]:
        """
        Compute gradients of a single coefficient w.r.t. shape parameters.
        
        This method supports:
        - Standard gradient computation (default)
        - VJP (vector-Jacobian product) when v is provided
        - JVP (Jacobian-vector product) when tangent is provided
        
        Parameters
        ----------
        coeff : str
            Coefficient to differentiate ("CL", "CD", or "CM").
        alpha : float
            Angle of attack in degrees.
        V : float
            Velocity in m/s.
        v : float or torch.Tensor, optional
            Upstream gradient scalar for VJP computation.
            If provided, returns v * dC/dp instead of dC/dp.
        tangent : list[torch.Tensor], optional
            Tangent vectors for JVP computation.
            Returns dot(dC/dp, tangent).
        return_dict : bool, default=False
            If True, return dict with keys "grads", "vjp", "jvp".
            If False (default), return grads or VJP directly.
        
        Returns
        -------
        list[torch.Tensor] or dict
            Gradients w.r.t. shape parameters.
            If return_dict=False: returns VJP if v provided, else raw gradients.
            If return_dict=True: returns {"grads": [...], "vjp": [...], "jvp": scalar}.
        
        Examples
        --------
        >>> # Standard gradient
        >>> grads = aero.backward("CL", alpha=5.0, V=18.0)
        
        >>> # VJP for trajectory optimization
        >>> grads = aero.backward("CL", alpha=5.0, V=18.0, v=0.8)
        
        >>> # Full dict output
        >>> result = aero.backward("CL", alpha=5.0, V=18.0, v=0.8, return_dict=True)
        >>> print(result["vjp"])
        """
        coeff = coeff.upper()
        assert coeff in ["CL", "CD", "CM"], f"coeff must be 'CL', 'CD', or 'CM', got {coeff}"
        
        # Select the appropriate block
        block = {
            "CL": self.cl_block,
            "CD": self.cd_block,
            "CM": self.cm_block
        }[coeff]
        
        # Compute gradients using the block's backward method
        grads = block.backward(alpha, V)
        
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
    
    def backward_combined(
        self,
        alpha: float,
        V: float,
        v_CL: float = 0.0,
        v_CD: float = 0.0,
        v_CM: float = 0.0
    ) -> List[torch.Tensor]:
        """
        Compute combined VJP for all coefficients in a single pass.
        
        This is the recommended method for trajectory optimization, as it
        computes the full gradient contribution from all three coefficients
        efficiently in a single forward+backward pass.
        
        Mathematically computes:
            dL/dp = (v_CL)(dCL/dp) + (v_CD)(dCD/dp) + (v_CM)(dCM/dp)
        
        where v_CL, v_CD, v_CM are upstream gradients from the trajectory loss.
        
        Parameters
        ----------
        alpha : float
            Angle of attack in degrees.
        V : float
            Velocity in m/s.
        v_CL : float, default=0.0
            Upstream gradient ∂L/∂CL from trajectory optimizer.
        v_CD : float, default=0.0
            Upstream gradient ∂L/∂CD from trajectory optimizer.
        v_CM : float, default=0.0
            Upstream gradient ∂L/∂CM from trajectory optimizer.
        
        Returns
        -------
        list[torch.Tensor]
            Gradients of trajectory loss w.r.t. shape parameters.
        
        Examples
        --------
        >>> # Typical usage in trajectory optimization
        >>> grads_shape = aero.backward_combined(
        ...     alpha=state.alpha,
        ...     V=state.velocity,
        ...     v_CL=adjoint.dL_dCL,
        ...     v_CD=adjoint.dL_dCD,
        ...     v_CM=adjoint.dL_dCM
        ... )
        >>> # Use grads_shape to update shape parameters
        
        Notes
        -----
        This is more efficient than calling backward() three times because:
        1. Only one LLT solve is needed
        2. PyTorch automatically accumulates gradients from all three coefficients
        3. Avoids redundant computation of influence matrices
        """
        # Zero existing gradients
        for p in self.shape_params:
            if p.grad is not None:
                p.grad.zero_()
        
        # Convert inputs to tensors
        device = self.shape_params[0].device
        alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
        V_t = torch.tensor(V, dtype=torch.float32, device=device)
        
        # Forward pass through LLT model (returns dict with CL, CD, CM)
        aero_dict = self.llt_model(alpha_t, V_t)
        
        # Extract coefficients and sum if they are vectors
        CL = aero_dict["CL"]
        CD = aero_dict["CD"]
        CM = aero_dict["CM"]
        
        if isinstance(CL, torch.Tensor) and CL.ndim > 0:
            CL = CL.sum()
        if isinstance(CD, torch.Tensor) and CD.ndim > 0:
            CD = CD.sum()
        if isinstance(CM, torch.Tensor) and CM.ndim > 0:
            CM = CM.sum()
        
        # Weighted sum (this is the VJP operation)
        loss = v_CL * CL + v_CD * CD + v_CM * CM
        
        # Backward pass (accumulates gradients in self.shape_params[i].grad)
        loss.backward()
        
        # Extract and return gradients
        return [p.grad.detach().clone() if p.grad is not None else None 
                for p in self.shape_params]
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def get_shape_params(self) -> List[torch.Tensor]:
        """
        Get trainable shape parameters.
        
        Returns
        -------
        list[torch.Tensor]
            List of Kulfan parameter tensors with requires_grad=True.
        """
        return self.shape_params
    
    def __repr__(self) -> str:
        """String representation of the AeroBlock."""
        n_params = sum(p.numel() for p in self.shape_params)
        device = self.shape_params[0].device if self.shape_params else "unknown"
        return (
            f"AeroBlock(part='{self.part}', mode='{self.mode}', "
            f"n_params={n_params}, device='{device}')"
        )
