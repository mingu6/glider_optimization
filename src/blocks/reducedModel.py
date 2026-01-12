from ..blockBase import Block
from typing import override
from ..config import Config
from typing import Dict, Any
import torch
import logging
import matplotlib.pyplot as plt

class ReducedModel(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging

        self._precomputed = False
        self._deg = self.config.reducedModel.chebyshev_degree
        self._l2_reg = self.config.reducedModel.l2_reg

    def _scale(self, x, min_val, max_val):
        mask = ~((x <= max_val) & (x >= min_val))
        if mask.any():
            print("Out of bounds:", x[mask])
        assert not mask.any()

        return 2 * (x - min_val) / (max_val - min_val) - 1

    def _chebyshev_basis(self, alpha_scaled, Re_scaled):
        deg = self._deg
        B = alpha_scaled.shape[0]
        
        T_alpha = torch.zeros(B, deg+1, device=alpha_scaled.device)
        T_Re = torch.zeros(B, deg+1, device=Re_scaled.device)
        
        T_alpha[:,0] = 1; T_Re[:,0] = 1
        if deg >= 1:
            T_alpha[:,1] = alpha_scaled; T_Re[:,1] = Re_scaled
            
        for n in range(2, deg+1):
            T_alpha[:,n] = 2*alpha_scaled*T_alpha[:,n-1] - T_alpha[:,n-2]
            T_Re[:,n] = 2*Re_scaled*T_Re[:,n-1] - T_Re[:,n-2]
            
        return (T_alpha.unsqueeze(-1) * T_Re.unsqueeze(-2)).reshape(T_alpha.shape[0], -1)

    def _precompute_chebyshev(self, alpha_scaled, Re_scaled):
        self._cheb_X = self._chebyshev_basis(alpha_scaled, Re_scaled)
        X = self._cheb_X
        reg = self._l2_reg * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        self._normal_lhs = torch.linalg.solve(X.T @ X + reg, X.T)    

        self._precomputed = True
    
    def _ridge_solve(self, y):
        if y.dim() == 1:
            y = y.view(-1,1)
        return self._normal_lhs @ y

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        alpha = downstream_info["alpha"].reshape(-1)
        Re = downstream_info["Re"].reshape(-1)

        self.alpha_min, self.alpha_max = alpha.min(), alpha.max()
        self.Re_min, self.Re_max = Re.min(), Re.max()

        nfConfig = self.config.neuralFoilSampling
        alpha_scaled = self._scale(alpha, nfConfig.AoA_min, nfConfig.AoA_max)
        Re_scaled = self._scale(Re, nfConfig.Re_min, nfConfig.Re_max)

        if not self._precomputed:
            self._precompute_chebyshev(alpha_scaled, Re_scaled) # The alpha, Re grid is actually constant (in neuralFoilSampling), so we can precompute the Chebyshev basis

        X_cheb = self._cheb_X

        CL = downstream_info["CL"].reshape(-1,1)
        CD = downstream_info["CD"].reshape(-1,1)
        CM = downstream_info["CM"].reshape(-1,1)

        if self.config.io.debug and False:
            B = X_cheb.shape[0]
            val_idx = torch.randperm(B)[:int(0.1 * B)]
            train_idx = torch.tensor([i for i in range(B) if i not in val_idx], device=X_cheb.device)

            X_train, X_val = X_cheb[train_idx], X_cheb[val_idx]
            CL_train, CL_val = CL[train_idx], CL[val_idx]
            CD_train, CD_val = CD[train_idx], CD[val_idx]
            CM_train, CM_val = CM[train_idx], CM[val_idx]

            coeffs_CL = self._ridge_solve(CL_train)
            coeffs_CD = self._ridge_solve(CD_train)
            coeffs_CM = self._ridge_solve(CM_train)

            train_errs = {
                "CL": torch.mean(torch.abs(CL_train - X_train @ coeffs_CL)).item(),
                "CD": torch.mean(torch.abs(CD_train - X_train @ coeffs_CD)).item(),
                "CM": torch.mean(torch.abs(CM_train - X_train @ coeffs_CM)).item(),
            }

            val_errs = {
                "CL": torch.mean(torch.abs(CL_val - X_val @ coeffs_CL)).item(),
                "CD": torch.mean(torch.abs(CD_val - X_val @ coeffs_CD)).item(),
                "CM": torch.mean(torch.abs(CM_val - X_val @ coeffs_CM)).item(),
            }

            self.logger.debug(
                "Chebyshev errors:\n"
                f"  CL -> train: {train_errs['CL']:.6f}, val: {val_errs['CL']:.6f}\n"
                f"  CD -> train: {train_errs['CD']:.6f}, val: {val_errs['CD']:.6f}\n"
                f"  CM -> train: {train_errs['CM']:.6f}, val: {val_errs['CM']:.6f}"
            )

        else:
            coeffs_CL = self._ridge_solve(CL)
            coeffs_CD = self._ridge_solve(CD)
            coeffs_CM = self._ridge_solve(CM)      
            
        # propagate augmented lagrangian if present so downstream blocks can access it
        aug = downstream_info.get("augmented_lagrangian", 0.0)
        return {
            "phi_CL": coeffs_CL, 
            "phi_CD": coeffs_CD, 
            "phi_CM": coeffs_CM,
            "augmented_lagrangian": aug,
        }

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        dphi_dy = self._normal_lhs
        dJ_dphi = upstream_grads["dJ_dphi"]
                
        if dphi_dy.isnan().any():
            self.logger.critical(f"⚠️ NaN detected in ReducedModel backward dphi_dy")
            
        dJ_dy = dJ_dphi @ dphi_dy
                            
        return {"dJ_dy": dJ_dy}

