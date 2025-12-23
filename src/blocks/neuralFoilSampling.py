from ..blockBase import Block
from typing import override
from ..config import Config
from ..utils.cu_kulfan_airfoil import get_aero_from_kulfan_parameters_cuda
from typing import Dict, Any
import torch
from math import sqrt   
import logging
import numpy as np
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
        aoa_1d = torch.linspace(nfConfig.AoA_min, nfConfig.AoA_max, n_1d, device=self.device)
        re_1d  = torch.linspace(nfConfig.Re_min, nfConfig.Re_max, n_1d, device=self.device)
        aoa, re = torch.meshgrid(aoa_1d, re_1d, indexing="ij")
        self.alpha_batch = aoa.reshape(-1)
        self.Re_batch = re.reshape(-1)
        
        self.last_airfoil = None
        
        self.lambda_conf = torch.tensor(0., device=self.device, requires_grad=False)
        self.rho = nfConfig.rho
        self.min_confidence = nfConfig.min_confidence

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
    
        return {
            "alpha": self.alpha_batch,
            "Re": self.Re_batch,
            "CL": self._last_aero_coeff["CL"].detach(),
            "CD": self._last_aero_coeff["CD"].detach(),
            "CM": self._last_aero_coeff["CM"].detach(),
        }

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        CL = self._last_aero_coeff["CL"]
        CD = self._last_aero_coeff["CD"]
        CM = self._last_aero_coeff["CM"]
        conf = self._last_aero_coeff["analysis_confidence"]
        
        constraint = self.min_confidence - conf.mean() 
        constraint_violation = torch.relu(constraint)
        constraint_lagrangian = self.lambda_conf * constraint_violation + self.rho/2 * (constraint_violation**2)

        if constraint_violation.detach() > 0.1:
            self.logger.critical("⚠️ Large confidence violation detected. Training may become unstable")
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
            
        with torch.no_grad():
            self.lambda_conf += self.rho * constraint_violation.mean().detach()
            
        return {
            "dupper_params": grad[0] + grad_lagrangian[0],
            "dlower_params": grad[1] + grad_lagrangian[1],
            "dleading_edge_param": grad[2] + grad_lagrangian[2],
            "dTE_thickness_param": grad[3] + grad_lagrangian[3],
        }
