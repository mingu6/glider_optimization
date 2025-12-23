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
        self.objective = {}

        self._precomputed = False
        self._deg = self.config.reducedModel.chebyshev_degree
        self._l2_reg = self.config.reducedModel.l2_reg
        
        self._coeffs = {}

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
        return downstream_info
        alpha = downstream_info["alpha"].reshape(-1)
        Re = downstream_info["Re"].reshape(-1)

        self.alpha_min, self.alpha_max = alpha.min(), alpha.max()
        self.Re_min, self.Re_max = Re.min(), Re.max()

        alpha_scaled = self._scale(alpha, self.alpha_min, self.alpha_max)
        Re_scaled = self._scale(Re, self.Re_min, self.Re_max)

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

        self._coeffs = {"CL": coeffs_CL, "CD": coeffs_CD, "CM": coeffs_CM}

        self.objective = {
            "CL": torch.mean((X_cheb @ coeffs_CL - CL)**2),
            "CD": torch.mean((X_cheb @ coeffs_CD - CD)**2),
            "CM": torch.mean((X_cheb @ coeffs_CM - CM)**2)
        }
        
        def predict_fn(alpha_new, Re_new):
            alpha_new = alpha_new.reshape(-1)
            Re_new = Re_new.reshape(-1)
            alpha_s = self._scale(alpha_new, self.alpha_min, self.alpha_max)
            Re_s = self._scale(Re_new, self.Re_min, self.Re_max)

            B = alpha_s.shape[0]
            deg = self._deg

            T_alpha_new = torch.zeros(B, deg+1, device=alpha_s.device)
            T_Re_new = torch.zeros(B, deg+1, device=Re_s.device)

            T_alpha_new[:,0] = 1
            T_Re_new[:,0] = 1
            if deg >= 1:
                T_alpha_new[:,1] = alpha_s
                T_Re_new[:,1] = Re_s

            for n in range(2, deg+1):
                T_alpha_new[:,n] = 2*alpha_s*T_alpha_new[:,n-1] - T_alpha_new[:,n-2]
                T_Re_new[:,n] = 2*Re_s*T_Re_new[:,n-1] - T_Re_new[:,n-2]

            X_new = (T_alpha_new.unsqueeze(-1) * T_Re_new.unsqueeze(-2)).reshape(B, -1)

            return {k: X_new @ v for k,v in self._coeffs.items()}

        return {"reduced_dynamic_fn": predict_fn}

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        """Compute gradient and Hessian using precomputed derivatives"""       
        
        return {"dJ_dy": upstream_grads["dJ_df"]}
         
        alpha_stream = upstream_grads["alpha"]
        Re_stream = upstream_grads["Re"]
        
        alpha_scaled = self._scale(alpha_stream, self.alpha_min, self.alpha_max)
        Re_scaled = self._scale(Re_stream, self.Re_min, self.Re_max)
        
        X_cheb_stream = self._chebyshev_basis(alpha_scaled, Re_scaled)
                
        df_dphi = torch.block_diag(X_cheb_stream, X_cheb_stream, X_cheb_stream)
        dphi_dy = torch.block_diag(self._normal_lhs, self._normal_lhs, self._normal_lhs)
        df_dphi = X_cheb_stream
        dphi_dy = self._normal_lhs
        
        if df_dphi.isnan().any():
            self.logger.critical(f"⚠️ NaN detected in ReducedModel backward df_dphi")
        
        dJ_df = upstream_grads["dJ_df"]

        dJ_dphi = dJ_df @ df_dphi
                
        if dphi_dy.isnan().any():
            self.logger.critical(f"⚠️ NaN detected in ReducedModel backward dphi_dy")
            
        dJ_dy = dJ_dphi @ dphi_dy
        
        if df_dphi.isnan().any():
            self.logger.critical(f"⚠️ NaN detected in ReducedModel backward dJ_dy")
            
        return {"dJ_dy": dJ_dy}

        # H = torch.zeros(B, 2, 2, device=self._T_alpha.device)
        # H[:,0,0] = V_factor**2 * (self._d2T_Re.unsqueeze(-1) * self._T_alpha.unsqueeze(-2) * coeffs).sum(dim=(1,2))
        # H[:,1,1] = AoA_factor**2 * (self._d2T_alpha.unsqueeze(-1) * self._T_Re.unsqueeze(-2) * coeffs).sum(dim=(1,2))
        # H[:,0,1] = H[:,1,0] = V_factor*AoA_factor * (self._d2T_alphaRe * coeffs).sum(dim=(1,2))

        
