from ..blockBase import Block
from typing import override
from ..config import Config
from ..utils.cu_kulfan_airfoil import cuKulfanAirfoil
from typing import Dict, Any
import aerosandbox as asb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import warnings
import torch.nn as nn
import torch

warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")

class Airfoil(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        af_conf = self.config.airfoil

        self.upper_params = nn.Parameter(torch.tensor(af_conf.upper_initial_weights, dtype=torch.float32))
        self.lower_params = nn.Parameter(torch.tensor(af_conf.lower_initial_weights, dtype=torch.float32))
        self.leading_edge_param = nn.Parameter(torch.tensor([af_conf.leading_edge_weight], dtype=torch.float32))
        self.TE_thickness_param = nn.Parameter(torch.tensor([af_conf.TE_thickness], dtype=torch.float32))

        self.optimizer = torch.optim.Adam([self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param], lr=af_conf.lr)

        self.frames = []

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self.plot()
        return {
            "upper_weights": self.upper_params, 
            "lower_weights": self.lower_params, 
            "leading_edge_weight": self.leading_edge_param, 
            "TE_thickness": self.TE_thickness_param, 
        }

    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        self.optimizer.zero_grad()
        
        self.upper_params.grad = upstream_grads["dupper_params"]
        self.lower_params.grad = upstream_grads["dlower_params"]
        self.leading_edge_param.grad = upstream_grads["dleading_edge_param"]
        self.TE_thickness_param.grad = upstream_grads["dTE_thickness_param"]
        
        self.optimizer.step()

        with torch.no_grad():
            self.TE_thickness_param.clamp_(1e-4, 0.5)
            min_gap = 0.002
            
            self.upper_params.data = torch.maximum(
                self.upper_params.data,
                self.lower_params.data + min_gap
            )
        
        # Projection
        
        '''
        # 1) Bounds for stability
        self.current_upper_params = np.clip(self.current_upper_params, -2.0, 2.0)
        self.current_lower_params = np.clip(self.current_lower_params, -2.0, 2.0)

        # 2) Upper must stay above lower
        min_gap = 0.002
        self.current_upper_params = np.maximum(
            self.current_upper_params,
            self.current_lower_params + min_gap
        )

        # 3) Enforce minimum average thickness (volume surrogate)
        min_avg_thickness = 0.01
        avg_thickness = np.mean(self.current_upper_params - self.current_lower_params)
        if avg_thickness < min_avg_thickness:
            scale = min_avg_thickness / (avg_thickness + 1e-8)
            mid = 0.5 * (self.current_upper_params + self.current_lower_params)
            half = (self.current_upper_params - self.current_lower_params) * 0.5 * scale
            self.current_upper_params = mid + half
            self.current_lower_params = mid - half
        '''
            
        return {}

    def plot(self):
        airfoilConfig = self.config.airfoil
        airfoil = asb.KulfanAirfoil(
            name=self.config.io.run_name + "_airfoil",
            lower_weights=self.lower_params.detach().numpy(),
            upper_weights=self.upper_params.detach().numpy(),
            leading_edge_weight=self.leading_edge_param.detach().numpy(),
            TE_thickness=self.TE_thickness_param.detach().numpy(),
            N1=airfoilConfig.N1,
            N2=airfoilConfig.N2,
        )

        fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    
        x = np.reshape(np.array(airfoil.x()), -1)
        y = np.reshape(np.array(airfoil.y()), -1)
    
        ax.plot(airfoil.x(), y, ".-", color="#280887", zorder=11)
        ax.fill(x, y, color="#280887", alpha=0.2, zorder=10)
        
        ax.text(
            0.02, 0.95, f"{len(self.frames)}", 
            transform=ax.transAxes, 
            fontsize=24, 
            fontweight="bold", 
            color="red",
            va="top", 
            ha="left"
        )
        
        ax.set_aspect(1.0, adjustable="datalim")
        
        fig.canvas.draw()
        buf, (w, h) = fig.canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
        self.frames.append(frame)
        
        plt.close(fig)


    def save_gif(self, filename="airfoil_evolution.gif", fps=10):
        if self.frames:
            imageio.mimsave(filename, self.frames, fps=fps)
