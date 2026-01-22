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
        self.device = torch.device(config.run.device)
                
        n_1d = int(sqrt(nfConfig.n_samples))
        aoa_1d = torch.linspace(nfConfig.AoA_min+1, nfConfig.AoA_max-1, n_1d, device=self.device)
        re_1d  = torch.linspace(nfConfig.Re_min+100, nfConfig.Re_max-100, n_1d, device=self.device)
        aoa, re = torch.meshgrid(aoa_1d, re_1d, indexing="ij")
        self.alpha_batch = aoa.reshape(-1)
        self.Re_batch = re.reshape(-1)
        
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        #fn = downstream_info["reduced_dynamic_fn"]

        #pred = fn(self.alpha_batch, self.Re_batch)

        self.last_CL = downstream_info["CL"].unsqueeze(0)  
        self.last_CD = downstream_info["CD"].unsqueeze(0)

        obj = -self.last_CL/self.last_CD
        
        return {"objective": obj.mean()}
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        dCL = - 1.0 / self.last_CD        
        dCL /= self.last_CD.shape[1]
        
        dCD = self.last_CL / (self.last_CD * self.last_CD)
        dCD /= self.last_CD.shape[1]
        
        dCM = torch.zeros_like(dCL)        
        
        grad = torch.cat([dCL, dCD, dCM])
                
        return {
            "dJ_dy": grad,
            "alpha": self.alpha_batch,
            "Re": self.Re_batch
        }
        