from ..blockBase import Block
from typing_extensions import override
from ..config import Config
from ..utils.cu_kulfan_airfoil import get_aero_from_kulfan_parameters_cuda
from ..utils.llt import build_llt_system
from typing import Dict, Any
import torch
from math import sqrt   
import logging
import numpy as np
import wandb
from ..utils.llt import LLTImplicitFn, _MODEL_SIZE_TO_ID, _DEVICE_TO_ID
from ..utils.spanwise_geometry import build_half_wing_stations_from_cfg, mix_root_tip_torch


class NeuralFoilSampling3D(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.device = torch.device(config.run.device)
        nfConfig = self.config.neuralFoilSampling
       
        n_span_stations = 7
        plane_cfg = getattr(self.config, "plane", {}) or {}
        wing_cfg = plane_cfg.get("wing", {}) if isinstance(plane_cfg, dict) else {}
        stations = build_half_wing_stations_from_cfg(wing_cfg, n_span_stations=n_span_stations)
        y_half = stations["y_half"].tolist()
        c_half = stations["c_half"].tolist()
        xle_half = stations["xle_half"].tolist()
        twist_half = stations["twist_half"].tolist()

        dihedral_deg = float(wing_cfg.get("dihedral", 0.0)) if isinstance(wing_cfg, dict) else 0.0
        comp = build_llt_system(y_half, c_half, xle_half, twist_half, dihedral_deg=dihedral_deg)

        self.llt_dy      = torch.as_tensor(comp["dy"],        dtype=torch.float32, device = self.device)
        self.llt_y       = torch.as_tensor(comp["y_mid"],     dtype=torch.float32, device = self.device)
        self.llt_c       = torch.as_tensor(comp["c_mid"],     dtype=torch.float32, device = self.device)
        self.llt_tw      = torch.as_tensor(comp["tw_mid"],    dtype=torch.float32, device = self.device)
        self.llt_S       = torch.as_tensor(comp["S"],         dtype=torch.float32, device = self.device)
        self.llt_cbar    = torch.as_tensor(comp["cbar"],      dtype=torch.float32, device = self.device)
        self.llt_x_c4    = torch.as_tensor(comp["x_c4_mid"],  dtype=torch.float32, device = self.device)
        self.llt_span    = torch.as_tensor(comp["span"],      dtype=torch.float32, device = self.device)
        self.llt_D_nf    = torch.as_tensor(comp["D_nf"],      dtype=torch.float32, device = self.device)
        self.llt_D_tr    = torch.as_tensor(comp["D_tr"],      dtype=torch.float32, device = self.device)
        self.llt_mirror  = torch.as_tensor(comp["mirror_of"], dtype=torch.long,    device = self.device)
        self.llt_cos_sw  = torch.as_tensor(comp["cos_sweep"],  dtype=torch.float32, device = self.device)
        self.llt_cos2_sw = torch.as_tensor(comp["cos2_sweep"], dtype=torch.float32, device = self.device)

        half_span = (self.llt_span * 0.5).clamp_min(1e-9)
        self.llt_eta = (self.llt_y.abs() / half_span).clamp(0.0, 1.0)
        self.llt_rho = torch.as_tensor(1.225, dtype=torch.float32, device = self.device)
        self.llt_mu = torch.as_tensor(1.789e-5, dtype=torch.float32, device = self.device)
        self.llt_model_size_id = torch.tensor(_MODEL_SIZE_TO_ID[self.config.neuralFoilSampling.neuralFoil_size], dtype=torch.int64, device = self.device)
        self.llt_device_id = torch.tensor(_DEVICE_TO_ID[config.run.device], dtype=torch.int64, device = self.device)
        
        
        beta = float(getattr(nfConfig, "llt_beta", 0.30))
        tol = float(getattr(nfConfig, "llt_tol", 1e-4))
        n_iter = int(getattr(nfConfig, "llt_n_iter", 20))
        max_iter = int(getattr(nfConfig, "llt_max_iter", 30))
        enforce_sym = True

        self.llt_beta_t = torch.tensor(beta, dtype=torch.float32)
        self.llt_tol_t = torch.tensor(tol, dtype=torch.float32)
        self.llt_n_iter_t = torch.tensor(float(n_iter), dtype=torch.float32)
        self.llt_max_iter_t = torch.tensor(float(max_iter), dtype=torch.float32)
        self.llt_enforce_sym_t = torch.tensor(1.0 if enforce_sym else 0.0, dtype=torch.float32)
        
        def chebyshev_nodes(a, b, n):
            k = np.arange(n)
            return 0.5*(a+b) + 0.5*(b-a)*np.cos((2*k+1)/(2*n)*np.pi)

        n_1d = int(sqrt(nfConfig.n_samples))
        aoa_1d = torch.tensor(chebyshev_nodes(nfConfig.AoA_min, nfConfig.AoA_max, n_1d), dtype=torch.float32, device=self.device)
        re_1d  = torch.tensor(chebyshev_nodes(nfConfig.Re_min, nfConfig.Re_max, n_1d), dtype=torch.float32, device=self.device)
        
        aoa, re = torch.meshgrid(aoa_1d, re_1d, indexing="ij")
        self.alpha_batch = aoa.reshape(-1)
        self.Re_batch = re.reshape(-1)
        
        # Validation set (random uniform)
        n_val = int(nfConfig.n_samples * 0.2) # 20% validation size
        self.alpha_val = (torch.rand(n_val, device=self.device) * (nfConfig.AoA_max - nfConfig.AoA_min)) + nfConfig.AoA_min
        self.Re_val = (torch.rand(n_val, device=self.device) * (nfConfig.Re_max - nfConfig.Re_min)) + nfConfig.Re_min
        
        self.last_airfoil = None
        
        self.lambda_conf = torch.tensor(0., device=self.device, requires_grad=False)
        self.rho = nfConfig.rho
        self.min_confidence = nfConfig.min_confidence
        
        self.min_avg_Cl_Cd = nfConfig.min_avg_Cl_Cd
        self.lambda_clcd = torch.tensor(0., device=self.device, requires_grad=False)

    def _eval_3d_llt(
        self,
        upper,
        lower,
        LE,
        TE,
        alpha_deg: torch.Tensor,
        Re_ref: torch.Tensor,
        upper_tip=None,
        lower_tip=None,
        LE_tip=None,
        TE_tip=None,
    ):
        """
        Minimal 3D wrapper: LLTImplicitFn expects (alpha, V).
        We map Re_ref -> V via V = Re_ref * mu / (rho * cbar).
        """

        eta = self.llt_eta  # (n_pan,)
        upper = mix_root_tip_torch(upper, upper_tip, eta)
        lower = mix_root_tip_torch(lower, lower_tip, eta)
        LE = (1.0 - eta) * LE.reshape(-1)[0] + eta * LE_tip.reshape(-1)[0]
        TE = (1.0 - eta) * TE.reshape(-1)[0] + eta * TE_tip.reshape(-1)[0]

        V = Re_ref * (self.llt_mu / (self.llt_rho * self.llt_cbar))

        C, alpha_eff_pan, Re_pan = LLTImplicitFn.apply(
            alpha_deg.reshape(-1), V.reshape(-1),
            upper, lower, LE.reshape(-1), TE.reshape(-1),
            self.llt_dy, self.llt_y, self.llt_c, self.llt_tw, self.llt_S, self.llt_cbar,
            self.llt_x_c4, self.llt_span,
            self.llt_D_nf, self.llt_D_tr, self.llt_mirror,
            self.llt_cos_sw, self.llt_cos2_sw,
            self.llt_rho, self.llt_mu,
            self.llt_beta_t, self.llt_tol_t, self.llt_n_iter_t, self.llt_max_iter_t, self.llt_enforce_sym_t,
            self.llt_model_size_id, self.llt_device_id,
        )

        # --- Per-panel confidence (spanwise) via one extra cuNeuralFoil call ---
        B, n_pan = alpha_eff_pan.shape
        BN = B * n_pan

        alpha_flat = alpha_eff_pan.reshape(-1)
        Re_flat = Re_pan.reshape(-1)

        # Build per-query Kulfan batch, matching implicit_llt._eval_nf_batched broadcasting rules
        if upper.ndim == 1:
            upper_batch = upper.unsqueeze(0).expand(BN, -1)
            lower_batch = lower.unsqueeze(0).expand(BN, -1)

            LE0 = LE.reshape(-1)[0]
            TE0 = TE.reshape(-1)[0]
            LE_batch = LE0.expand(BN)
            TE_batch = TE0.expand(BN)

        elif upper.ndim == 2:
            # (n_pan, 8) -> (B, n_pan, 8) -> (BN, 8)
            upper_batch = upper.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)
            lower_batch = lower.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)

            LEv = LE.reshape(-1)
            TEv = TE.reshape(-1)
            if LEv.numel() == 1:
                LE_pan = LEv[0].expand(n_pan)
            else:
                LE_pan = LEv
            if TEv.numel() == 1:
                TE_pan = TEv[0].expand(n_pan)
            else:
                TE_pan = TEv

            LE_batch = LE_pan.unsqueeze(0).expand(B, -1).reshape(BN)
            TE_batch = TE_pan.unsqueeze(0).expand(B, -1).reshape(BN)

        else:
            raise ValueError(f"Unexpected upper.ndim={upper.ndim} for Kulfan weights")

        kulfan_conf = {
            "upper_weights_cuda": upper_batch,
            "lower_weights_cuda": lower_batch,
            "leading_edge_weight_cuda": LE_batch,
            "TE_thickness_cuda": TE_batch,
        }

        aero_conf = get_aero_from_kulfan_parameters_cuda(
            kulfan_conf,
            alpha_flat,
            Re_flat,
            device=self.device,
            model_size=self.config.neuralFoilSampling.neuralFoil_size,
        )
        conf_flat = aero_conf.get("analysis_confidence", torch.ones_like(alpha_flat))
        conf_pan = conf_flat.view(B, n_pan)

        # Aggregate conservatively 
        # conf_min = conf_pan.min(dim=1).values  # (B,)
        conf_mean = conf_pan.mean(dim=1)

        return {"CL": C[:, 0], "CD": C[:, 1], "CM": C[:, 2], "analysis_confidence": conf_mean}

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self._last_input = downstream_info
        self._current_iteration = downstream_info["iteration"]
        
        self._last_aero_coeff = self._eval_3d_llt(
                downstream_info["upper_weights"],
                downstream_info["lower_weights"],
                downstream_info["leading_edge_weight"],
                downstream_info["TE_thickness"],
                self.alpha_batch,
                self.Re_batch,
                upper_tip=downstream_info["upper_weights_tip"],
                lower_tip=downstream_info["lower_weights_tip"],
                LE_tip=downstream_info["leading_edge_weight_tip"],
                TE_tip=downstream_info["TE_thickness_tip"],
            )   
   
        conf = self._last_aero_coeff.get("analysis_confidence")
        try:
            conf_mean = float(conf.mean().detach().cpu().item())
        except Exception:
            conf_mean = 1.0

        constraint_violation = max(0.0, self.min_confidence - conf_mean)
        lambda_val = float(self.lambda_conf.detach().cpu().item()) if isinstance(self.lambda_conf, torch.Tensor) else float(self.lambda_conf)
        aug_lagrangian = lambda_val * constraint_violation + 0.5 * float(self.rho) * (constraint_violation ** 2)

        # Cl/Cd constraint
        CL_fwd = self._last_aero_coeff["CL"].detach()
        CD_fwd = self._last_aero_coeff["CD"].detach()
        CD_safe_fwd = torch.clamp(CD_fwd, min=1e-5)
        cl_cd_mean = float((CL_fwd / CD_safe_fwd).mean().cpu().item())
        
        violation_clcd = max(0.0, self.min_avg_Cl_Cd - cl_cd_mean)
        lambda_clcd_val = float(self.lambda_clcd.detach().cpu().item()) if isinstance(self.lambda_clcd, torch.Tensor) else float(self.lambda_clcd)
        aug_lagrangian += lambda_clcd_val * violation_clcd + 0.5 * float(self.rho) * (violation_clcd ** 2)
        
        if self.config.io.wandb.enabled:
            metrics = {
                "lagrangian/lambda_conf": lambda_val,
                "lagrangian/lambda_clcd": lambda_clcd_val
            }
            wandb.log(metrics, step=downstream_info["iteration"])

        # Validation forward pass
        B_val = self.alpha_val.shape[0]
        kulfan_batch_val = {
            "upper_weights_cuda": downstream_info["upper_weights"].repeat(B_val, 1),
            "lower_weights_cuda": downstream_info["lower_weights"].repeat(B_val, 1),
            "leading_edge_weight_cuda": downstream_info["leading_edge_weight"].repeat(B_val),
            "TE_thickness_cuda": downstream_info["TE_thickness"].repeat(B_val),
        }

        val_aero = get_aero_from_kulfan_parameters_cuda(
            kulfan_batch_val,
            self.alpha_val,
            self.Re_val,
            device=self.device,
            model_size=self.config.neuralFoilSampling.neuralFoil_size,
        )

        out = {
            "alpha": self.alpha_batch,
            "Re": self.Re_batch,
            "CL": self._last_aero_coeff["CL"].detach(),
            "CD": self._last_aero_coeff["CD"].detach(),
            "CM": self._last_aero_coeff["CM"].detach(),
            "augmented_lagrangian": aug_lagrangian,
            # Validation data
            "val_alpha": self.alpha_val,
            "val_Re": self.Re_val,
            "val_CL": val_aero["CL"].detach(),
            "val_CD": val_aero["CD"].detach(),
            "val_CM": val_aero["CM"].detach(),
            "iteration": downstream_info["iteration"]
        }
        if "wing_reference_geometry" in downstream_info:
            out["wing_reference_geometry"] = downstream_info["wing_reference_geometry"]
        return out

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        CL = self._last_aero_coeff["CL"]
        CD = self._last_aero_coeff["CD"]
        CM = self._last_aero_coeff["CM"]
        conf = self._last_aero_coeff["analysis_confidence"]
        
        constraint = self.min_confidence - conf.mean() 
        constraint_violation = torch.relu(constraint)
        constraint_lagrangian = self.lambda_conf * constraint_violation + self.rho/2 * (constraint_violation**2)

        # Cl/Cd constraint backward
        CD_safe = torch.clamp(CD, min=1e-5)
        cl_cd_ratio = CL / CD_safe
        constraint_clcd = self.min_avg_Cl_Cd - cl_cd_ratio.mean()
        violation_clcd = torch.relu(constraint_clcd)
        constraint_lagrangian_clcd = self.lambda_clcd * violation_clcd + self.rho/2 * (violation_clcd**2)
        
        constraint_lagrangian = constraint_lagrangian + constraint_lagrangian_clcd
        
        if violation_clcd.detach() > 1.0:
             self.logger.warning(f"⚠️ Large Cl/Cd violation. Mean: {cl_cd_ratio.mean().detach():.3f}. Target: {self.min_avg_Cl_Cd:.3f}")

        if constraint_violation.detach() > 0.1:
            self.logger.critical(f"⚠️ Large confidence violation detected. Training may become unstable. Mean Confidence {conf.mean():.3f}. Target {self.min_confidence:.3f}")
        if CL.isnan().any():
            self.logger.critical("⚠️ NaN detected in NeuralFoilSampling feedforward CL")
        if CD.isnan().any():
            self.logger.critical("⚠️ NaN detected in NeuralFoilSampling feedforward CD")
        if CM.isnan().any():
            self.logger.critical("⚠️ NaN detected in NeuralFoilSampling feedforward CM")
                
        dJ_dy = upstream_grads["dJ_dy"]
        
        upper = self._last_input["upper_weights"]
        lower = self._last_input["lower_weights"]
        LE = self._last_input["leading_edge_weight"]
        TE = self._last_input["TE_thickness"]        
        upper_tip = self._last_input["upper_weights_tip"]
        lower_tip = self._last_input["lower_weights_tip"]
        LE_tip = self._last_input["leading_edge_weight_tip"]
        TE_tip = self._last_input["TE_thickness_tip"]
        
        params = [upper, lower, LE, TE, upper_tip, lower_tip, LE_tip, TE_tip]
                
        Y = torch.cat([CL, CD, CM], dim=0)
        
        grad_lagrangian = torch.autograd.grad(constraint_lagrangian, params, retain_graph = True)
        grad = torch.autograd.grad(Y, params, grad_outputs=dJ_dy.flatten())
        
        for i, g in enumerate(grad):
            if g.isnan().any():
                self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[{i}]")
            
        with torch.no_grad():
            self.lambda_conf += self.rho * constraint_violation.mean().detach()
            self.lambda_clcd += self.rho * violation_clcd.mean().detach()
            
        return {
            "dupper_params": grad[0] + grad_lagrangian[0],
            "dlower_params": grad[1] + grad_lagrangian[1],
            "dleading_edge_param": grad[2] + grad_lagrangian[2],
            "dTE_thickness_param": grad[3] + grad_lagrangian[3],
            "dupper_params_tip": grad[4] + grad_lagrangian[4],
            "dlower_params_tip": grad[5] + grad_lagrangian[5],
            "dleading_edge_param_tip": grad[6] + grad_lagrangian[6],
            "dTE_thickness_param_tip": grad[7] + grad_lagrangian[7],
        }
        
    def resume(self, checkpoint):
        self.lambda_conf = torch.tensor(checkpoint["lagrangian/lambda_conf"], device=self.device, dtype=torch.float32, requires_grad=False)
        self.lambda_clcd = torch.tensor(checkpoint["lagrangian/lambda_clcd"], device=self.device, dtype=torch.float32, requires_grad=False)

