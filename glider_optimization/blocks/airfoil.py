from ..blockBase import Block
from typing_extensions import override
from ..config import Config
from pathlib import Path
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
import torch.optim.lr_scheduler as lr_scheduler
import wandb

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

        self.optimizer = torch.optim.Adam(
            [self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param],
            lr=af_conf.lr
        )
        
        self._iter = 0
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=af_conf.gamma)
        self.frames = []

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self._iter = downstream_info["iteration"]

        if self._iter % self.config.io.log_every == 0:
            self.plot()
            if self.config.io.wandb.enabled:
                self._log_params_to_wandb()
                
        return {
            "upper_weights": self.upper_params,
            "lower_weights": self.lower_params,
            "leading_edge_weight": self.leading_edge_param,
            "TE_thickness": self.TE_thickness_param,
            "iteration": downstream_info["iteration"]
        }

    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        self._apply_gradients(upstream_grads)
        self.optimizer.step()
        self._step_scheduler()
        self._enforce_constraints()
        
        if self._iter == self.config.run.max_outer_iters - 1 and not self.config.io.wandb.enabled:
            self.save_gif(fps=self.config.io.gif_fps)
        
        return {}
    
    def resume(self, checkpoint):
        upper_weights = []
        lower_weights = []
        
        for i in range(8):
            upper_key = f"airfoil/upper_params_{i}"
            lower_key = f"airfoil/lower_params_{i}"
            upper_weights.append(checkpoint[upper_key])
            lower_weights.append(checkpoint[lower_key])            
                    
        leading_edge_weight = float(checkpoint["airfoil/leading_edge_weight"])
        TE_thickness = float(checkpoint["airfoil/TE_thickness"])
        
        self.upper_params = nn.Parameter(torch.tensor(upper_weights, dtype=torch.float32))
        self.lower_params = nn.Parameter(torch.tensor(lower_weights, dtype=torch.float32))
        self.leading_edge_param = nn.Parameter(torch.tensor([leading_edge_weight], dtype=torch.float32))
        self.TE_thickness_param = nn.Parameter(torch.tensor([TE_thickness], dtype=torch.float32))
        self.optimizer = torch.optim.Adam(
            [self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param],
            lr=self.config.airfoil.lr
        )
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.airfoil.gamma)


    def get_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return float(getattr(self.config.airfoil, "lr", 0.0))

    def _log_params_to_wandb(self):
        metrics = {"airfoil/learning_rate": self.get_lr()}
        
        for i, val in enumerate(self.upper_params.detach().numpy()):
            metrics[f"airfoil/upper_params_{i}"] = float(val)
        for i, val in enumerate(self.lower_params.detach().numpy()):
            metrics[f"airfoil/lower_params_{i}"] = float(val)
        
        metrics["airfoil/leading_edge_weight"] = float(self.leading_edge_param.detach().numpy()[0])
        metrics["airfoil/TE_thickness"] = float(self.TE_thickness_param.detach().numpy()[0])
        
        wandb.log(metrics, step=self._iter)

    def _apply_gradients(self, upstream_grads):
        self.optimizer.zero_grad()
        self.upper_params.grad = upstream_grads["dupper_params"]
        self.lower_params.grad = upstream_grads["dlower_params"]
        self.leading_edge_param.grad = upstream_grads["dleading_edge_param"]
        self.TE_thickness_param.grad = upstream_grads["dTE_thickness_param"]

    def _step_scheduler(self):
        if self.scheduler is None:
            return

        try:
            self.scheduler.step()
        except Exception:
            warnings.warn("LR scheduler step failed; continuing without scheduling.")

    def _enforce_constraints(self):
        with torch.no_grad():
            self.TE_thickness_param.clamp_(1e-4, 0.01)
            min_gap = 0.05
            self.upper_params.data = torch.maximum(
                self.upper_params.data,
                self.lower_params.data + min_gap
            )

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
        
        ax.axis("off")
        ax.set_aspect(1.0, adjustable="datalim")
        
        fig.canvas.draw()
        buf, (w, h) = fig.canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
        
        if self.config.io.wandb.enabled:
            wandb.log({"airfoil/shape": wandb.Image(frame, caption=f"Airfoil Iter {self._iter}")}, step=self._iter)
        else:
            self.frames.append(frame)
        
        plt.close(fig)


    def save_gif(self, filename="airfoil_evolution.gif", fps=1):        
        if self.frames:
            log_dir = Path(self.config.io.checkpoint_dir)
            imageio.mimsave(log_dir/filename, self.frames, fps=fps)
