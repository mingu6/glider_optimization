from ..blockBase import Block
from typing import override
from ..config import Config
from typing import Dict, Any
from math import sqrt
import torch
import logging
class Evaluation(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        nfConfig = self.config.neuralFoilSampling
        self.logger = logging
                
        n_1d = int(sqrt(nfConfig.n_samples))
        aoa_1d = torch.linspace(nfConfig.AoA_min, nfConfig.AoA_max, n_1d)
        re_1d  = torch.linspace(nfConfig.Re_min, nfConfig.Re_max, n_1d)
        aoa, re = torch.meshgrid(aoa_1d, re_1d, indexing="ij")
        self.alpha_batch = aoa.reshape(-1)
        self.Re_batch = re.reshape(-1)
        
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        fn = downstream_info["reduced_dynamic_fn"]

        pred = fn(self.alpha_batch, self.Re_batch)

        self.last_CL = pred["CL"][:,0].unsqueeze(0)  
        self.last_CD = pred["CD"][:,0].unsqueeze(0)

        obj = -self.last_CL/self.last_CD
        return {"objective": obj.mean()}
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        dCL = - 1.0 / self.last_CD        
        dCL /= self.last_CD.shape[1]
        dCD = self.last_CL / (self.last_CD * self.last_CD)
        dCD /= self.last_CD.shape[1]
        
        grad = torch.cat([dCL, dCD, torch.zeros_like(dCL)])
        
        self.logger.info("Config")
        self.logger.info(grad.abs().mean().item())
        
        return {
            "dJ_df": grad,
            "alpha": self.alpha_batch,
            "Re": self.Re_batch
        }
        