from ..blockBase import Block
from typing import override
from ..config import Config
from typing import Dict, Any
import torch

class Evaluation(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        fn = downstream_info["reduced_dynamic_fn"]
        
        self.alpha_new = torch.tensor([-5.0])
        self.Re_new    = torch.tensor([1.0e6])

        pred = fn(self.alpha_new, self.Re_new)

        self.last_CL = pred["CL"][0,0]  
        self.last_CD = pred["CD"][0,0]

        obj = self.last_CL / self.last_CD

        return {"objective": obj}
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        dCL = 1.0 / self.last_CD
        dCD = -(self.last_CL / (self.last_CD * self.last_CD))
        return {
            "dJ_df": torch.tensor([dCL, dCD, 0]),
            "alpha": self.alpha_new,
            "Re": self.Re_new
        }