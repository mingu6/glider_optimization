from ..blockBase import Block
from typing import override
from ..config import Config
from ..utils.cu_kulfan_airfoil import cuKulfanAirfoil
from typing import Dict, Any
import aerosandbox as asb
import matplotlib.pyplot as plt

class Airfoil(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.current_upper_params = self.config.airfoil.upper_initial_weights
        self.current_lower_params = self.config.airfoil.lower_initial_weights
                
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        airfoilConfig = self.config.airfoil
        
        airfoil = asb.KulfanAirfoil(
            name = self.config.io.run_name + "_airfoil",
            lower_weights = self.current_lower_params,
            upper_weights = self.current_upper_params,
            leading_edge_weight = airfoilConfig.leading_edge_weight,
            TE_thickness = airfoilConfig.TE_thickness,
            N1 = airfoilConfig.N1,
            N2 = airfoilConfig.N2,
        )
        cuairfoil = cuKulfanAirfoil(airfoil, requires_grad = True, device = self.config.run.device)
        return {
            "airfoil":cuairfoil
        }
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        alpha = 0.00001
        dJ_dw = upstream_grads["dJ_dw"]
        dJ_dw_normalized = dJ_dw 
        dJ_dw_normalized = dJ_dw_normalized.detach().numpy()[0]
        self.current_upper_params -= alpha * dJ_dw_normalized[:8]
        self.current_lower_params -= alpha * dJ_dw_normalized[8:16]
        
        return {}
    
    def plot(self):
        airfoilConfig = self.config.airfoil
        
        airfoil = asb.KulfanAirfoil(
            name = self.config.io.run_name + "_airfoil",
            lower_weights = self.current_lower_params,
            upper_weights = self.current_upper_params,
            leading_edge_weight = airfoilConfig.leading_edge_weight,
            TE_thickness = airfoilConfig.TE_thickness,
            N1 = airfoilConfig.N1,
            N2 = airfoilConfig.N2,
        )
        fig, ax = plt.subplots(figsize=(6, 2))
        airfoil.draw()