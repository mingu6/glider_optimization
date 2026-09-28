from ..blockBase import Block
from typing import override
from ..config import Config
from ..utils.cu_kulfan_airfoil import get_aero_from_kulfan_parameters_cuda
from ..utils.augmented_lagrangian import AugmentedLagrangian
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

        # Chebyshev nodes cluster at the ends of the interval -- correct for the
        # polynomial fit these samples feed, but wrong for a constraint written as a
        # mean: it put 37.5% of the weight beyond |AoA| 25 deg, where NeuralFoil is
        # unconfident for any airfoil, so the constraint was decided by a regime the
        # glider never flies in. Chebyshev-Gauss quadrature says the uniform-measure
        # average is recovered by weighting each node with sqrt(1 - x^2) in the
        # normalised coordinate; the sampling itself is left untouched.
        def quadrature_weights(nodes_1d, lo, hi):
            x = 2.0 * (nodes_1d - lo) / (hi - lo) - 1.0
            return torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))

        w_aoa = quadrature_weights(aoa_1d, nfConfig.AoA_min, nfConfig.AoA_max)
        w_re = quadrature_weights(re_1d, nfConfig.Re_min, nfConfig.Re_max)
        w = torch.outer(w_aoa, w_re).reshape(-1)
        self.constraint_weights = (w / w.sum()).to(torch.float32)
        
        n_val = int(nfConfig.n_samples * 0.2) # 20% validation size
        self.alpha_val = (torch.rand(n_val, device=self.device) * (nfConfig.AoA_max - nfConfig.AoA_min)) + nfConfig.AoA_min
        self.Re_val = (torch.rand(n_val, device=self.device) * (nfConfig.Re_max - nfConfig.Re_min)) + nfConfig.Re_min
        
        self.last_airfoil = None
        
        self.min_confidence = nfConfig.min_confidence
        self.min_avg_Cl_Cd = nfConfig.min_avg_Cl_Cd
        self.augmented_lagrangian = AugmentedLagrangian(
            names=("conf", "clcd"),
            rho=nfConfig.rho,
            rho_max=nfConfig.rho_max,
            rho_factor=nfConfig.rho_factor,
            infeasibility_decrease=nfConfig.infeasibility_decrease,
            penalty_every=nfConfig.penalty_every,
            dual_every=nfConfig.dual_every,
        )

    def _envelope_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Average over the (AoA, Re) envelope under a uniform measure."""
        return (values * self.constraint_weights.to(values.device)).sum()

    def constraints(self, aero_coeff: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Design constraints, in the g(x) <= 0 form the augmented Lagrangian expects."""
        cl_cd = aero_coeff["CL"] / torch.clamp(aero_coeff["CD"], min=1e-5)
        return {
            "conf": self.min_confidence - self._envelope_mean(aero_coeff["analysis_confidence"]),
            "clcd": self.min_avg_Cl_Cd - self._envelope_mean(cl_cd),
        }

    def _log_constraint_warnings(self, constraints: Dict[str, torch.Tensor]):
        clcd, conf = (float(constraints[k].detach()) for k in ("clcd", "conf"))
        if clcd > 1.0:
            self.logger.warning(
                f"⚠️ Large Cl/Cd violation. Deficit: {clcd:.3f}. Target: {self.min_avg_Cl_Cd:.3f}"
            )
        if conf > 0.1:
            self.logger.critical(
                f"⚠️ Large confidence violation detected. Training may become unstable. "
                f"Deficit {conf:.3f}. Target {self.min_confidence:.3f}"
            )

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        B = self.alpha_batch.shape[0]
        
        self._last_input = downstream_info
        
        kulfan_batch = {
            "upper_weights_cuda": downstream_info["upper_weights"].repeat(B, 1),
            "lower_weights_cuda": downstream_info["lower_weights"].repeat(B, 1),
            "leading_edge_weight_cuda": downstream_info["leading_edge_weight"].repeat(B),
            "TE_thickness_cuda": downstream_info["TE_thickness"].repeat(B),
        }

        self._last_aero_coeff = get_aero_from_kulfan_parameters_cuda(
            kulfan_batch,
            self.alpha_batch,
            self.Re_batch,
            device=self.device,
            model_size=self.config.neuralFoilSampling.neuralFoil_size,
        )
        
        constraints = {k: v.detach() for k, v in self.constraints(self._last_aero_coeff).items()}
        aug_lagrangian = float(self.augmented_lagrangian.penalty(constraints).cpu().item())

        if self.config.io.wandb.enabled:
            wandb.log(self.augmented_lagrangian.metrics(constraints), step=downstream_info["iteration"])

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
            "iteration": downstream_info["iteration"]
        }

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        CL = self._last_aero_coeff["CL"]
        CD = self._last_aero_coeff["CD"]
        CM = self._last_aero_coeff["CM"]

        constraints = self.constraints(self._last_aero_coeff)
        constraint_lagrangian = self.augmented_lagrangian.penalty(constraints)

        self._log_constraint_warnings(constraints)

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
            
        self.augmented_lagrangian.step({k: float(v.detach()) for k, v in constraints.items()})

        return {
            "dupper_params": grad[0] + grad_lagrangian[0],
            "dlower_params": grad[1] + grad_lagrangian[1],
            "dleading_edge_param": grad[2] + grad_lagrangian[2],
            "dTE_thickness_param": grad[3] + grad_lagrangian[3],
        }
        
    def resume(self, checkpoint):
        self.augmented_lagrangian.load_state_dict(checkpoint)

