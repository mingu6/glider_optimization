from ..blockBase import Block
from typing_extensions import override
from ..config import Config
from ..utils.cu_kulfan_airfoil import get_aero_from_kulfan_parameters_cuda
from typing import Dict, Any
import torch
from math import sqrt   
import logging
import numpy as np
import wandb

class NeuralFoilSampling(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.device = torch.device(config.run.device)
        nfConfig = self.config.neuralFoilSampling
        
        def chebyshev_nodes(a, b, n):
            k = np.arange(n)
            return 0.5*(a+b) + 0.5*(b-a)*np.cos((2*k+1)/(2*n)*np.pi)

        n_1d = int(sqrt(nfConfig.n_samples))
        aoa_1d = torch.tensor(chebyshev_nodes(nfConfig.AoA_min, nfConfig.AoA_max, n_1d), device=self.device)
        re_1d  = torch.tensor(chebyshev_nodes(nfConfig.Re_min, nfConfig.Re_max, n_1d), device=self.device)
        
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

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        B = self.alpha_batch.shape[0]
        
        self._last_input = downstream_info
        
        kulfan_batch = {
            "upper_weights_cuda": downstream_info["upper_weights"].detach().repeat(B, 1),
            "lower_weights_cuda": downstream_info["lower_weights"].detach().repeat(B, 1),
            "leading_edge_weight_cuda": downstream_info["leading_edge_weight"].detach().repeat(B),
            "TE_thickness_cuda": downstream_info["TE_thickness"].detach().repeat(B),
        }
   
        # Run in no_grad: backward() will re-run with gradients when needed.
        # This avoids building a massive autograd graph through the xxxlarge network
        # for all B samples during the forward pass (same pattern as LLTImplicitFn).
        with torch.no_grad():
            self._last_aero_coeff = get_aero_from_kulfan_parameters_cuda(
                kulfan_batch,
                self.alpha_batch,
                self.Re_batch,
                device=self.device,
                model_size=self.config.neuralFoilSampling.neuralFoil_size,
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

        # Validation forward pass (no_grad: outputs are detached below)
        B_val = self.alpha_val.shape[0]
        kulfan_batch_val = {
            "upper_weights_cuda": downstream_info["upper_weights"].detach().repeat(B_val, 1),
            "lower_weights_cuda": downstream_info["lower_weights"].detach().repeat(B_val, 1),
            "leading_edge_weight_cuda": downstream_info["leading_edge_weight"].detach().repeat(B_val),
            "TE_thickness_cuda": downstream_info["TE_thickness"].detach().repeat(B_val),
        }

        with torch.no_grad():
            val_aero = get_aero_from_kulfan_parameters_cuda(
                kulfan_batch_val,
                self.alpha_val,
                self.Re_val,
                device=self.device,
                model_size=self.config.neuralFoilSampling.neuralFoil_size,
            )

        return {
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
            "iteration": downstream_info["iteration"],
            "wing_reference_geometry": downstream_info.get("wing_reference_geometry"),
        }

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        upper = self._last_input["upper_weights"]
        lower = self._last_input["lower_weights"]
        LE = self._last_input["leading_edge_weight"]
        TE = self._last_input["TE_thickness"]
        B = self.alpha_batch.shape[0]

        # Re-run forward pass with gradients enabled for differentiation.
        kulfan_batch_bwd = {
            "upper_weights_cuda": upper.repeat(B, 1),
            "lower_weights_cuda": lower.repeat(B, 1),
            "leading_edge_weight_cuda": LE.repeat(B),
            "TE_thickness_cuda": TE.repeat(B),
        }
        aero = get_aero_from_kulfan_parameters_cuda(
            kulfan_batch_bwd,
            self.alpha_batch,
            self.Re_batch,
            device=self.device,
            model_size=self.config.neuralFoilSampling.neuralFoil_size,
        )
        CL = aero["CL"]
        CD = aero["CD"]
        CM = aero["CM"]
        conf = aero["analysis_confidence"]

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
                
        Y = torch.cat([CL, CD, CM], dim=0)
        
        grad_lagrangian = torch.autograd.grad(constraint_lagrangian, [upper, lower, LE, TE], retain_graph = True )
        grad = torch.autograd.grad(Y, [upper, lower, LE, TE], grad_outputs=dJ_dy.flatten())
        
        if grad[0].isnan().any():
            self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[0]")
        if grad[1].isnan().any():
            self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[1]")
        if grad[2].isnan().any():
            self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[2]")
        if grad[3].isnan().any():
            self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[3]")
            
        with torch.no_grad():
            self.lambda_conf += self.rho * constraint_violation.mean().detach()
            self.lambda_clcd += self.rho * violation_clcd.mean().detach()
            
        return {
            "dupper_params": grad[0] + grad_lagrangian[0],
            "dlower_params": grad[1] + grad_lagrangian[1],
            "dleading_edge_param": grad[2] + grad_lagrangian[2],
            "dTE_thickness_param": grad[3] + grad_lagrangian[3],
        }
        
    def resume(self, checkpoint):
        self.lambda_conf = torch.tensor(checkpoint["lagrangian/lambda_conf"], device=self.device, dtype=torch.float32, requires_grad=False)
        self.lambda_clcd = torch.tensor(checkpoint["lagrangian/lambda_clcd"], device=self.device, dtype=torch.float32, requires_grad=False)

