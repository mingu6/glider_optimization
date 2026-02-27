from ..blockBase import Block
from typing import override
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
import logging
from ..utils.airfoil_debug import log_kulfan_parameters, log_backward_update, is_airfoil_debug_enabled

warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")

class Airfoil3D(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        af_conf = self.config.airfoil
        self.device = torch.device(config.run.device)
        
        self.upper_params = nn.Parameter(torch.tensor(af_conf.upper_initial_weights, dtype=torch.float32))
        self.lower_params = nn.Parameter(torch.tensor(af_conf.lower_initial_weights, dtype=torch.float32))
        self.leading_edge_param = nn.Parameter(torch.tensor([af_conf.leading_edge_weight], dtype=torch.float32))
        self.TE_thickness_param = nn.Parameter(torch.tensor([af_conf.TE_thickness], dtype=torch.float32))
        
        # self.upper_params_tip = nn.Parameter(torch.tensor(af_conf.upper_initial_weights, dtype=torch.float32))
        # self.lower_params_tip = nn.Parameter(torch.tensor(af_conf.lower_initial_weights, dtype=torch.float32))
        # self.leading_edge_param_tip = nn.Parameter(torch.tensor([af_conf.leading_edge_weight], dtype=torch.float32))
        # self.TE_thickness_param_tip = nn.Parameter(torch.tensor([af_conf.TE_thickness], dtype=torch.float32))

        tip_upper = af_conf.upper_initial_weights_tip if af_conf.upper_initial_weights_tip is not None else af_conf.upper_initial_weights
        tip_lower = af_conf.lower_initial_weights_tip if af_conf.lower_initial_weights_tip is not None else af_conf.lower_initial_weights
        tip_le = af_conf.leading_edge_weight_tip if af_conf.leading_edge_weight_tip is not None else af_conf.leading_edge_weight
        tip_te = af_conf.TE_thickness_tip if af_conf.TE_thickness_tip is not None else af_conf.TE_thickness

        self.upper_params_tip = nn.Parameter(torch.tensor(tip_upper, dtype=torch.float32))
        self.lower_params_tip = nn.Parameter(torch.tensor(tip_lower, dtype=torch.float32))
        self.leading_edge_param_tip = nn.Parameter(torch.tensor([tip_le], dtype=torch.float32))
        self.TE_thickness_param_tip = nn.Parameter(torch.tensor([tip_te], dtype=torch.float32))
        # with torch.no_grad():
        #     self.upper_params_tip[0].add_(1e-4)

        self.optimizer = torch.optim.Adam(
            [
                self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param,
                self.upper_params_tip, self.lower_params_tip, self.leading_edge_param_tip, self.TE_thickness_param_tip,
            ],
            lr=af_conf.lr
        )
        
        self._iter = 0
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=af_conf.gamma)
        self.frames = []

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self._iter = downstream_info["iteration"]

        if is_airfoil_debug_enabled() and (self._iter % self.config.io.log_every == 0):
            log_kulfan_parameters(
                iteration=self._iter,
                stage="iteration_start",
                logger=self.logger,
                checkpoint_dir=self.config.io.checkpoint_dir,
                root_upper=self.upper_params,
                root_lower=self.lower_params,
                root_le=self.leading_edge_param,
                root_te=self.TE_thickness_param,
                tip_upper=self.upper_params_tip,
                tip_lower=self.lower_params_tip,
                tip_le=self.leading_edge_param_tip,
                tip_te=self.TE_thickness_param_tip,
            )

        if self._iter % self.config.io.log_every == 0:
            self.plot()
            if self.config.io.wandb.enabled:
                self._log_params_to_wandb()
                
        return {
            "upper_weights": self.upper_params.to(self.device),
            "lower_weights": self.lower_params.to(self.device),
            "leading_edge_weight": self.leading_edge_param.to(self.device),
            "TE_thickness": self.TE_thickness_param.to(self.device),
            "upper_weights_tip": self.upper_params_tip.to(self.device),
            "lower_weights_tip": self.lower_params_tip.to(self.device),
            "leading_edge_weight_tip": self.leading_edge_param_tip.to(self.device),
            "TE_thickness_tip": self.TE_thickness_param_tip.to(self.device),
            "iteration": downstream_info["iteration"]
        }

    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        if is_airfoil_debug_enabled():
            before_root_upper = self.upper_params.detach().clone()
            before_root_lower = self.lower_params.detach().clone()
            before_root_le = self.leading_edge_param.detach().clone()
            before_root_te = self.TE_thickness_param.detach().clone()
            before_tip_upper = self.upper_params_tip.detach().clone()
            before_tip_lower = self.lower_params_tip.detach().clone()
            before_tip_le = self.leading_edge_param_tip.detach().clone()
            before_tip_te = self.TE_thickness_param_tip.detach().clone()

        self._apply_gradients(upstream_grads)

        if is_airfoil_debug_enabled():
            grad_root_upper = self.upper_params.grad.detach().clone() if self.upper_params.grad is not None else None
            grad_root_lower = self.lower_params.grad.detach().clone() if self.lower_params.grad is not None else None
            grad_root_le = self.leading_edge_param.grad.detach().clone() if self.leading_edge_param.grad is not None else None
            grad_root_te = self.TE_thickness_param.grad.detach().clone() if self.TE_thickness_param.grad is not None else None
            grad_tip_upper = self.upper_params_tip.grad.detach().clone() if self.upper_params_tip.grad is not None else None
            grad_tip_lower = self.lower_params_tip.grad.detach().clone() if self.lower_params_tip.grad is not None else None
            grad_tip_le = self.leading_edge_param_tip.grad.detach().clone() if self.leading_edge_param_tip.grad is not None else None
            grad_tip_te = self.TE_thickness_param_tip.grad.detach().clone() if self.TE_thickness_param_tip.grad is not None else None

        self.optimizer.step()
        self._step_scheduler()
        self._enforce_constraints()

        if is_airfoil_debug_enabled():
            log_backward_update(
                iteration=self._iter,
                checkpoint_dir=self.config.io.checkpoint_dir,
                before_root_upper=before_root_upper,
                before_root_lower=before_root_lower,
                before_root_le=before_root_le,
                before_root_te=before_root_te,
                before_tip_upper=before_tip_upper,
                before_tip_lower=before_tip_lower,
                before_tip_le=before_tip_le,
                before_tip_te=before_tip_te,
                grad_root_upper=grad_root_upper,
                grad_root_lower=grad_root_lower,
                grad_root_le=grad_root_le,
                grad_root_te=grad_root_te,
                grad_tip_upper=grad_tip_upper,
                grad_tip_lower=grad_tip_lower,
                grad_tip_le=grad_tip_le,
                grad_tip_te=grad_tip_te,
                after_root_upper=self.upper_params,
                after_root_lower=self.lower_params,
                after_root_le=self.leading_edge_param,
                after_root_te=self.TE_thickness_param,
                after_tip_upper=self.upper_params_tip,
                after_tip_lower=self.lower_params_tip,
                after_tip_le=self.leading_edge_param_tip,
                after_tip_te=self.TE_thickness_param_tip,
            )
        
        if not self.config.io.wandb.enabled:
            gif_every = max(1, int(getattr(self.config.io, "airfoil_gif_every", self.config.io.log_every)))
            is_final_iter = self._iter == self.config.run.max_outer_iters - 1
            if self._iter == 0 or ((self._iter + 1) % gif_every == 0) or is_final_iter:
                self.save_gif(fps=self.config.io.gif_fps)
        
        return {}
    
    def resume(self, checkpoint):
        upper_weights = []
        lower_weights = []
        upper_weights_tip = []
        lower_weights_tip = []

        for i in range(8):
            upper_key = f"airfoil/upper_params_{i}"
            lower_key = f"airfoil/lower_params_{i}"
            upper_weights.append(checkpoint[upper_key])
            lower_weights.append(checkpoint[lower_key])

            upper_key_tip = f"airfoil/upper_params_{i}_tip"
            lower_key_tip = f"airfoil/lower_params_{i}_tip"
            upper_weights_tip.append(checkpoint[upper_key_tip] if upper_key_tip in checkpoint else checkpoint[upper_key])
            lower_weights_tip.append(checkpoint[lower_key_tip] if lower_key_tip in checkpoint else checkpoint[lower_key])

        leading_edge_weight = float(checkpoint["airfoil/leading_edge_weight"])
        TE_thickness = float(checkpoint["airfoil/TE_thickness"])
        leading_edge_weight_tip = float(checkpoint.get("airfoil/leading_edge_weight_tip", leading_edge_weight))
        TE_thickness_tip = float(checkpoint.get("airfoil/TE_thickness_tip", TE_thickness))

        self.upper_params = nn.Parameter(torch.tensor(upper_weights, dtype=torch.float32))
        self.lower_params = nn.Parameter(torch.tensor(lower_weights, dtype=torch.float32))
        self.leading_edge_param = nn.Parameter(torch.tensor([leading_edge_weight], dtype=torch.float32))
        self.TE_thickness_param = nn.Parameter(torch.tensor([TE_thickness], dtype=torch.float32))

        self.upper_params_tip = nn.Parameter(torch.tensor(upper_weights_tip, dtype=torch.float32))
        self.lower_params_tip = nn.Parameter(torch.tensor(lower_weights_tip, dtype=torch.float32))
        self.leading_edge_param_tip = nn.Parameter(torch.tensor([leading_edge_weight_tip], dtype=torch.float32))
        self.TE_thickness_param_tip = nn.Parameter(torch.tensor([TE_thickness_tip], dtype=torch.float32))

        self.optimizer = torch.optim.Adam(
            [
                self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param,
                self.upper_params_tip, self.lower_params_tip, self.leading_edge_param_tip, self.TE_thickness_param_tip,
            ],
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
        
        for i, val in enumerate(self.upper_params_tip.detach().numpy()):
            metrics[f"airfoil/upper_params_{i}_tip"] = float(val)
        for i, val in enumerate(self.lower_params_tip.detach().numpy()):
            metrics[f"airfoil/lower_params_{i}_tip"] = float(val)
        
        metrics["airfoil/leading_edge_weight_tip"] = float(self.leading_edge_param_tip.detach().numpy()[0])
        metrics["airfoil/TE_thickness_tip"] = float(self.TE_thickness_param_tip.detach().numpy()[0])
        
        wandb.log(metrics, step=self._iter)

    def _apply_gradients(self, upstream_grads):
        self.optimizer.zero_grad()
        self.upper_params.grad = upstream_grads["dupper_params"]
        self.lower_params.grad = upstream_grads["dlower_params"]
        self.leading_edge_param.grad = upstream_grads["dleading_edge_param"]
        self.TE_thickness_param.grad = upstream_grads["dTE_thickness_param"]
        self.upper_params_tip.grad = upstream_grads["dupper_params_tip"]
        self.lower_params_tip.grad = upstream_grads["dlower_params_tip"]
        self.leading_edge_param_tip.grad = upstream_grads["dleading_edge_param_tip"]
        self.TE_thickness_param_tip.grad = upstream_grads["dTE_thickness_param_tip"]

    def _step_scheduler(self):
        if self.scheduler is None:
            return

        try:
            self.scheduler.step()
        except Exception:
            warnings.warn("LR scheduler step failed; continuing without scheduling.")

    def _enforce_constraints(self):
        with torch.no_grad():
            self.TE_thickness_param.clamp_(0, 0.01)
            self.TE_thickness_param_tip.clamp_(0, 0.01)
            
            min_gap = 0.05
            
            self.upper_params.data = torch.maximum(
                self.upper_params.data,
                self.lower_params.data + min_gap
            )
            self.upper_params_tip.data = torch.maximum(
                self.upper_params_tip.data,
                self.lower_params_tip.data + min_gap
            )

    def plot(self):
        airfoilConfig = self.config.airfoil
        airfoil_root = asb.KulfanAirfoil(
            name=self.config.io.run_name + "_airfoil_root",
            lower_weights=self.lower_params.detach().numpy(),
            upper_weights=self.upper_params.detach().numpy(),
            leading_edge_weight=self.leading_edge_param.detach().numpy(),
            TE_thickness=self.TE_thickness_param.detach().numpy(),
            N1=airfoilConfig.N1,
            N2=airfoilConfig.N2,
        )

        airfoil_tip = asb.KulfanAirfoil(
            name=self.config.io.run_name + "_airfoil_tip",
            lower_weights=self.lower_params_tip.detach().numpy(),
            upper_weights=self.upper_params_tip.detach().numpy(),
            leading_edge_weight=self.leading_edge_param_tip.detach().numpy(),
            TE_thickness=self.TE_thickness_param_tip.detach().numpy(),
            N1=airfoilConfig.N1,
            N2=airfoilConfig.N2,
        )

        fig, ax = plt.subplots(figsize=(6, 3), dpi=200)

        x_root = np.reshape(np.array(airfoil_root.x()), -1)
        y_root = np.reshape(np.array(airfoil_root.y()), -1)
        x_tip = np.reshape(np.array(airfoil_tip.x()), -1)
        y_tip = np.reshape(np.array(airfoil_tip.y()), -1)

        ax.plot(x_root, y_root, ".-", color="#280887", zorder=11, label="Root")
        ax.fill(x_root, y_root, color="#280887", alpha=0.2, zorder=10)

        ax.plot(x_tip, y_tip, ".-", color="#d97706", zorder=13, label="Tip")
        ax.fill(x_tip, y_tip, color="#d97706", alpha=0.15, zorder=12)

        ax.legend(loc="upper right", frameon=False)
        
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
