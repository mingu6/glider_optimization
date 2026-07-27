from pathlib import Path
from typing import Any, Dict, override

import aerosandbox as asb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import warnings
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import wandb

from ..blockBase import Block
from ..config import Config
from ..utils.spanwise_geometry import dynamic_centroid_offset

warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")


class Airfoil3D(Block):
    """Root+tip Kulfan airfoil design variables for the spanwise (LLT) pipeline.

    Root and tip start from the same initial shape (the config only specifies one
    airfoil) but are optimized as independent parameters, so the wing is free to
    develop a spanwise taper in shape if that helps.
    """

    @override
    def __init__(self, config: Config):
        self.config = config
        af_conf = self.config.airfoil

        def param(value):
            return nn.Parameter(torch.tensor(value, dtype=torch.float32))

        self.upper_params = param(af_conf.upper_initial_weights)
        self.lower_params = param(af_conf.lower_initial_weights)
        self.leading_edge_param = param([af_conf.leading_edge_weight])
        self.TE_thickness_param = param([af_conf.TE_thickness])

        self.upper_params_tip = param(af_conf.upper_initial_weights)
        self.lower_params_tip = param(af_conf.lower_initial_weights)
        self.leading_edge_param_tip = param([af_conf.leading_edge_weight])
        self.TE_thickness_param_tip = param([af_conf.TE_thickness])

        self.optimizer = torch.optim.Adam(
            [
                self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param,
                self.upper_params_tip, self.lower_params_tip, self.leading_edge_param_tip, self.TE_thickness_param_tip,
            ],
            lr=af_conf.lr,
        )
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=af_conf.gamma)

        self._iter = 0
        self.frames = []
        self._n_span_stations = 7

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self._iter = downstream_info["iteration"]

        if self._iter % self.config.io.log_every == 0:
            self.plot()
            if self.config.io.wandb.enabled:
                self._log_params_to_wandb()

        out = {
            "upper_weights": self.upper_params,
            "lower_weights": self.lower_params,
            "leading_edge_weight": self.leading_edge_param,
            "TE_thickness": self.TE_thickness_param,
            "upper_weights_tip": self.upper_params_tip,
            "lower_weights_tip": self.lower_params_tip,
            "leading_edge_weight_tip": self.leading_edge_param_tip,
            "TE_thickness_tip": self.TE_thickness_param_tip,
            "iteration": downstream_info["iteration"],
        }

        wing_cfg = (getattr(self.config, "plane", {}) or {}).get("wing", {})
        if wing_cfg.get("dynamic_centroid", False):
            out["wing_centroid_offset"] = dynamic_centroid_offset(
                wing_cfg,
                self._root_kulfan(),
                self._tip_kulfan(),
                n_span_stations=self._n_span_stations,
            )
        return out

    def _root_kulfan(self) -> Dict[str, Any]:
        return self._kulfan_dict(self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param)

    def _tip_kulfan(self) -> Dict[str, Any]:
        return self._kulfan_dict(self.upper_params_tip, self.lower_params_tip, self.leading_edge_param_tip, self.TE_thickness_param_tip)

    @staticmethod
    def _kulfan_dict(upper, lower, le, te) -> Dict[str, Any]:
        return {
            "upper_weights": upper.detach().cpu().numpy(),
            "lower_weights": lower.detach().cpu().numpy(),
            "leading_edge_weight": float(le.detach().cpu().numpy()[0]),
            "TE_thickness": float(te.detach().cpu().numpy()[0]),
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
        def load_vector(prefix):
            return torch.tensor([checkpoint[f"airfoil/{prefix}_{i}"] for i in range(8)], dtype=torch.float32)

        def load_scalar(key):
            return torch.tensor([float(checkpoint[f"airfoil/{key}"])], dtype=torch.float32)

        self.upper_params = nn.Parameter(load_vector("upper_params"))
        self.lower_params = nn.Parameter(load_vector("lower_params"))
        self.leading_edge_param = nn.Parameter(load_scalar("leading_edge_weight"))
        self.TE_thickness_param = nn.Parameter(load_scalar("TE_thickness"))

        self.upper_params_tip = nn.Parameter(load_vector("upper_params_tip"))
        self.lower_params_tip = nn.Parameter(load_vector("lower_params_tip"))
        self.leading_edge_param_tip = nn.Parameter(load_scalar("leading_edge_weight_tip"))
        self.TE_thickness_param_tip = nn.Parameter(load_scalar("TE_thickness_tip"))

        self.optimizer = torch.optim.Adam(
            [
                self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param,
                self.upper_params_tip, self.lower_params_tip, self.leading_edge_param_tip, self.TE_thickness_param_tip,
            ],
            lr=self.config.airfoil.lr,
        )
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.airfoil.gamma)

    def get_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return float(getattr(self.config.airfoil, "lr", 0.0))

    def _log_params_to_wandb(self):
        metrics = {"airfoil/learning_rate": self.get_lr()}

        def log_vector(prefix, tensor):
            for i, val in enumerate(tensor.detach().cpu().numpy()):
                metrics[f"airfoil/{prefix}_{i}"] = float(val)

        log_vector("upper_params", self.upper_params)
        log_vector("lower_params", self.lower_params)
        log_vector("upper_params_tip", self.upper_params_tip)
        log_vector("lower_params_tip", self.lower_params_tip)
        metrics["airfoil/leading_edge_weight"] = float(self.leading_edge_param.item())
        metrics["airfoil/TE_thickness"] = float(self.TE_thickness_param.item())
        metrics["airfoil/leading_edge_weight_tip"] = float(self.leading_edge_param_tip.item())
        metrics["airfoil/TE_thickness_tip"] = float(self.TE_thickness_param_tip.item())

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
        try:
            self.scheduler.step()
        except Exception:
            warnings.warn("LR scheduler step failed; continuing without scheduling.")

    def _enforce_constraints(self):
        with torch.no_grad():
            self.TE_thickness_param.clamp_(1e-4, 0.01)
            self.TE_thickness_param_tip.clamp_(1e-4, 0.01)
            min_gap = 0.05
            self.upper_params.data = torch.maximum(self.upper_params.data, self.lower_params.data + min_gap)
            self.upper_params_tip.data = torch.maximum(self.upper_params_tip.data, self.lower_params_tip.data + min_gap)

    def plot(self):
        af_conf = self.config.airfoil
        root = asb.KulfanAirfoil(
            name=self.config.io.run_name + "_root",
            upper_weights=self.upper_params.detach().cpu().numpy(),
            lower_weights=self.lower_params.detach().cpu().numpy(),
            leading_edge_weight=self.leading_edge_param.detach().cpu().numpy(),
            TE_thickness=self.TE_thickness_param.detach().cpu().numpy(),
            N1=af_conf.N1, N2=af_conf.N2,
        )
        tip = asb.KulfanAirfoil(
            name=self.config.io.run_name + "_tip",
            upper_weights=self.upper_params_tip.detach().cpu().numpy(),
            lower_weights=self.lower_params_tip.detach().cpu().numpy(),
            leading_edge_weight=self.leading_edge_param_tip.detach().cpu().numpy(),
            TE_thickness=self.TE_thickness_param_tip.detach().cpu().numpy(),
            N1=af_conf.N1, N2=af_conf.N2,
        )

        x_root, y_root = np.reshape(np.array(root.x()), -1), np.reshape(np.array(root.y()), -1)
        x_tip, y_tip = np.reshape(np.array(tip.x()), -1), np.reshape(np.array(tip.y()), -1)

        fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
        ax.plot(x_root, y_root, ".-", color="#280887", zorder=11, label="root")
        ax.fill(x_root, y_root, color="#280887", alpha=0.2, zorder=10)
        ax.plot(x_tip, y_tip, ".-", color="#d97706", zorder=13, label="tip")
        ax.fill(x_tip, y_tip, color="#d97706", alpha=0.15, zorder=12)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
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
            imageio.mimsave(log_dir / filename, self.frames, fps=fps)
