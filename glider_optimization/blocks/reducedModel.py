from ..blockBase import Block
from typing import override
from ..config import Config
from typing import Dict, Any
import torch
import logging


class ReducedModel(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging

        self._precomputed = False
        self._deg = self.config.reducedModel.chebyshev_degree
        self._l2_reg = self.config.reducedModel.l2_reg
        self._cheb_X = None
        self._normal_lhs = None

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        alpha = downstream_info["alpha"].reshape(-1)
        Re = downstream_info["Re"].reshape(-1)

        self.alpha_min, self.alpha_max = alpha.min(), alpha.max()
        self.Re_min, self.Re_max = Re.min(), Re.max()

        nfConfig = self.config.neuralFoilSampling
        alpha_scaled = self._scale_to_domain(alpha, nfConfig.AoA_min, nfConfig.AoA_max)
        Re_scaled = self._scale_to_domain(Re, nfConfig.Re_min, nfConfig.Re_max)

        if not self._precomputed:
            self._precompute_chebyshev(alpha_scaled, Re_scaled)

        CL = downstream_info["CL"].reshape(-1, 1)
        CD = downstream_info["CD"].reshape(-1, 1)
        CM = downstream_info["CM"].reshape(-1, 1)

        coeffs_CL = self._ridge_solve(CL)
        coeffs_CD = self._ridge_solve(CD)
        coeffs_CM = self._ridge_solve(CM)

        # Optional fixed-elevator surrogate (computed once in NeuralFoilSampling in 3D mode)
        if "CL_e" in downstream_info and "CD_e" in downstream_info and "CM_e" in downstream_info:
            CL_e = downstream_info["CL_e"].reshape(-1, 1)
            CD_e = downstream_info["CD_e"].reshape(-1, 1)
            CM_e = downstream_info["CM_e"].reshape(-1, 1)
            coeffs_CL_e = self._ridge_solve(CL_e)
            coeffs_CD_e = self._ridge_solve(CD_e)
            coeffs_CM_e = self._ridge_solve(CM_e)
        else:
            coeffs_CL_e = coeffs_CD_e = coeffs_CM_e = None

        if "val_alpha" in downstream_info:
            self._validate_model(downstream_info, coeffs_CL, coeffs_CD, coeffs_CM, nfConfig)

        aug = downstream_info.get("augmented_lagrangian", 0.0)
        return {
            "phi_CL": coeffs_CL,
            "phi_CD": coeffs_CD,
            "phi_CM": coeffs_CM,
            **({
                "phi_CL_e": coeffs_CL_e,
                "phi_CD_e": coeffs_CD_e,
                "phi_CM_e": coeffs_CM_e,
            } if coeffs_CL_e is not None else {}), #Return elevator phi_*_e only if they exist, otherwise don’t include those keys at all.
            "augmented_lagrangian": aug,
            "iteration": downstream_info["iteration"]
        }

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        dphi_dy = self._normal_lhs
        dJ_dphi = upstream_grads["dJ_dphi"]
        
        # Only the wing airfoil is optimized. If the OCP provides additional
        # blocks (e.g. elevator), ignore them here so gradients match the wing
        # samples (CL, CD, CM) only.
        if dJ_dphi.shape[0] > 3:
            dJ_dphi = dJ_dphi[:3, :]

        if dphi_dy.isnan().any():
            self.logger.critical(f"⚠️ NaN detected in ReducedModel backward dphi_dy")
        
        dJ_dy = dJ_dphi @ dphi_dy
        return {"dJ_dy": dJ_dy}

    def _scale_to_domain(self, x, min_val, max_val):
        mask = ~((x <= max_val) & (x >= min_val))
        if mask.any():
            print("Out of bounds:", x[mask])
        assert not mask.any()
        return 2 * (x - min_val) / (max_val - min_val) - 1

    def _chebyshev_basis(self, alpha_scaled, Re_scaled):
        deg = self._deg
        B = alpha_scaled.shape[0]
        
        T_alpha = torch.zeros(B, deg + 1, device=alpha_scaled.device)
        T_Re = torch.zeros(B, deg + 1, device=Re_scaled.device)
        
        T_alpha[:, 0] = 1
        T_Re[:, 0] = 1
        
        if deg >= 1:
            T_alpha[:, 1] = alpha_scaled
            T_Re[:, 1] = Re_scaled
        
        for n in range(2, deg + 1):
            T_alpha[:, n] = 2 * alpha_scaled * T_alpha[:, n - 1] - T_alpha[:, n - 2]
            T_Re[:, n] = 2 * Re_scaled * T_Re[:, n - 1] - T_Re[:, n - 2]
        
        return (T_alpha.unsqueeze(-1) * T_Re.unsqueeze(-2)).reshape(T_alpha.shape[0], -1)

    def _precompute_chebyshev(self, alpha_scaled, Re_scaled):
        self._cheb_X = self._chebyshev_basis(alpha_scaled, Re_scaled)
        X = self._cheb_X
        reg = self._l2_reg * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        self._normal_lhs = torch.linalg.solve(X.T @ X + reg, X.T)
        self._precomputed = True

    def _ridge_solve(self, y):
        if y.dim() == 1:
            y = y.view(-1, 1)
        return self._normal_lhs @ y

    def _validate_model(self, downstream_info, coeffs_CL, coeffs_CD, coeffs_CM, nfConfig):
        val_alpha = downstream_info["val_alpha"]
        val_Re = downstream_info["val_Re"]
        val_CL = downstream_info["val_CL"].reshape(-1, 1)
        val_CD = downstream_info["val_CD"].reshape(-1, 1)
        val_CM = downstream_info["val_CM"].reshape(-1, 1)

        val_alpha_scaled = self._scale_to_domain(val_alpha, nfConfig.AoA_min, nfConfig.AoA_max)
        val_Re_scaled = self._scale_to_domain(val_Re, nfConfig.Re_min, nfConfig.Re_max)
        
        X_val = self._chebyshev_basis(val_alpha_scaled, val_Re_scaled)
        
        pred_CL = X_val @ coeffs_CL
        pred_CD = X_val @ coeffs_CD
        pred_CM = X_val @ coeffs_CM

        val_errs = {
            "CL_mse": torch.mean((val_CL - pred_CL) ** 2).item(),
            "CD_mse": torch.mean((val_CD - pred_CD) ** 2).item(),
            "CM_mse": torch.mean((val_CM - pred_CM) ** 2).item(),
        }

        threshold = 1e-2
        if val_errs['CL_mse'] > threshold:
            self.logger.critical(f"⚠️ CL validation error too high: {val_errs['CL_mse']:.2e}")
        if val_errs['CD_mse'] > threshold:
            self.logger.critical(f"⚠️ CD validation error too high: {val_errs['CD_mse']:.2e}")
        if val_errs['CM_mse'] > threshold:
            self.logger.critical(f"⚠️ CM validation error too high: {val_errs['CM_mse']:.2e}")
