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

warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")

class Airfoil(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        af_conf = self.config.airfoil

        # Optional: optimize both root and tip airfoils (used only by 3D LLT branch).
        # 3D-only spanwise wing airfoil root/tip optimization can be seeded either from:
        # - plane.wing.kulfan_root / kulfan_tip (preferred when provided), or
        # - plane.wing.airfoil_root / airfoil_tip names (legacy fallback).
        use_3d = bool(getattr(self.config, "neuralFoilSampling", None) and self.config.neuralFoilSampling.use_3d_llt)
        wing_cfg = getattr(getattr(self.config, "plane", None), "wing", None)

        wing_has_tip = bool(
            wing_cfg is not None
            and (
                getattr(wing_cfg, "airfoil_tip", None) is not None
                or getattr(wing_cfg, "kulfan_tip", None) is not None
            )
        )
        self.spanwise_enabled = bool(use_3d and wing_has_tip)

        # self.upper_params = nn.Parameter(torch.tensor(af_conf.upper_initial_weights, dtype=torch.float32))
        # self.lower_params = nn.Parameter(torch.tensor(af_conf.lower_initial_weights, dtype=torch.float32))
        # self.leading_edge_param = nn.Parameter(torch.tensor([af_conf.leading_edge_weight], dtype=torch.float32))
        # self.TE_thickness_param = nn.Parameter(torch.tensor([af_conf.TE_thickness], dtype=torch.float32))

        def _from_named_airfoil(name_raw: str):
            name = str(name_raw).lower().replace("_", "").replace(" ", "")
            k = asb.Airfoil(name).to_kulfan_airfoil()
            upper = np.array(k.upper_weights, dtype=float)
            lower = np.array(k.lower_weights, dtype=float)
            le = float(k.leading_edge_weight)
            te = float(k.TE_thickness)
            if upper.shape[0] != 8 or lower.shape[0] != 8:
                raise ValueError(
                    f"Expected 8 Kulfan weights from airfoil '{name_raw}', got upper={upper.shape}, lower={lower.shape}"
                )
            return upper, lower, le, te

        def _from_kulfan_spec(spec):
            upper = np.array(spec.upper_weights, dtype=float)
            lower = np.array(spec.lower_weights, dtype=float)
            le = float(spec.leading_edge_weight)
            te = float(spec.TE_thickness)
            if upper.shape[0] != 8 or lower.shape[0] != 8:
                raise ValueError(
                    f"Kulfan spec must have 8 weights for upper/lower; got upper={upper.shape}, lower={lower.shape}"
                )
            return upper, lower, le, te

        # ROOT initialization:
        # - 2D: from top-level YAML Kulfan weights (unchanged default behavior)
        # - 3D: prefer plane.wing.kulfan_root (or plane.wing.kulfan),
        #       fallback to plane.wing.airfoil_root (or plane.wing.airfoil)
        root_upper_init = np.array(af_conf.upper_initial_weights, dtype=float)
        root_lower_init = np.array(af_conf.lower_initial_weights, dtype=float)
        root_le_init = float(af_conf.leading_edge_weight)
        root_te_init = float(af_conf.TE_thickness)

        if use_3d and wing_cfg is not None:
            root_kulfan = getattr(wing_cfg, "kulfan_root", None) or getattr(wing_cfg, "kulfan", None)
            if root_kulfan is not None:
                root_upper_init, root_lower_init, root_le_init, root_te_init = _from_kulfan_spec(root_kulfan)
            else:
                root_name_raw = getattr(wing_cfg, "airfoil_root", None) or getattr(wing_cfg, "airfoil", None)
                if root_name_raw is not None:
                    root_upper_init, root_lower_init, root_le_init, root_te_init = _from_named_airfoil(root_name_raw)

        self.upper_params = nn.Parameter(torch.tensor(root_upper_init, dtype=torch.float32))
        self.lower_params = nn.Parameter(torch.tensor(root_lower_init, dtype=torch.float32))
        self.leading_edge_param = nn.Parameter(torch.tensor([root_le_init], dtype=torch.float32))
        self.TE_thickness_param = nn.Parameter(torch.tensor([root_te_init], dtype=torch.float32))

        params_for_optim = [self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param]

        # Tip parameters (only created/optimized when spanwise is enabled)
        if self.spanwise_enabled:
            # Tip initialization: prefer plane.wing.kulfan_tip, fallback to plane.wing.airfoil_tip
            tip_kulfan = getattr(wing_cfg, "kulfan_tip", None)
            if tip_kulfan is not None:
                tip_upper_init, tip_lower_init, tip_le_init, tip_te_init = _from_kulfan_spec(tip_kulfan)
            else:
                tip_name_raw = getattr(wing_cfg, "airfoil_tip", None)
                if tip_name_raw is None:
                    raise ValueError("spanwise_enabled=True requires plane.wing.airfoil_tip or plane.wing.kulfan_tip")
                tip_upper_init, tip_lower_init, tip_le_init, tip_te_init = _from_named_airfoil(tip_name_raw)

            self.upper_params_tip = nn.Parameter(torch.tensor(tip_upper_init, dtype=torch.float32))
            self.lower_params_tip = nn.Parameter(torch.tensor(tip_lower_init, dtype=torch.float32))
            self.leading_edge_param_tip = nn.Parameter(torch.tensor([tip_le_init], dtype=torch.float32))
            self.TE_thickness_param_tip = nn.Parameter(torch.tensor([tip_te_init], dtype=torch.float32))

            params_for_optim += [self.upper_params_tip, self.lower_params_tip, self.leading_edge_param_tip, self.TE_thickness_param_tip]

        self.optimizer = torch.optim.Adam(params_for_optim, lr=af_conf.lr)

        # self.optimizer = torch.optim.Adam(
        #     [self.upper_params, self.lower_params, self.leading_edge_param, self.TE_thickness_param],
        #     lr=af_conf.lr
        # )
        
        self._iter = 0
        self.scheduler = self._create_scheduler(af_conf)
        self.frames = []

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self._iter = downstream_info.get("iteration", self._iter)

        if getattr(self.config.io, "log_airfoil_coeffs", True):
            self._log_airfoil_coeffs()

        if self._iter % self.config.io.log_every == 0:
            self.plot()
            if self.config.io.wandb.enabled:
                self._log_params_to_wandb()
        
        # return {
        #     "upper_weights": self.upper_params,
        #     "lower_weights": self.lower_params,
        #     "leading_edge_weight": self.leading_edge_param,
        #     "TE_thickness": self.TE_thickness_param,
        #     "iteration": downstream_info["iteration"]
        # }

        out = {
            "upper_weights": self.upper_params,
            "lower_weights": self.lower_params,
            "leading_edge_weight": self.leading_edge_param,
            "TE_thickness": self.TE_thickness_param,
            "iteration": downstream_info["iteration"],
        }

        # Optional tip outputs (3D LLT only)
        if self.spanwise_enabled:
            out.update({
                "upper_weights_tip": self.upper_params_tip,
                "lower_weights_tip": self.lower_params_tip,
                "leading_edge_weight_tip": self.leading_edge_param_tip,
                "TE_thickness_tip": self.TE_thickness_param_tip,
            })

        return out

    def _log_airfoil_coeffs(self):
        root_upper = np.array(self.upper_params.detach().cpu().numpy(), dtype=float)
        root_lower = np.array(self.lower_params.detach().cpu().numpy(), dtype=float)
        root_le = float(self.leading_edge_param.detach().cpu().numpy()[0])
        root_te = float(self.TE_thickness_param.detach().cpu().numpy()[0])

        self.logger.info(
            "Iter %d | airfoil root | upper=%s | lower=%s | LE=%.6f | TE=%.6f",
            self._iter,
            np.array2string(root_upper, precision=5, separator=", "),
            np.array2string(root_lower, precision=5, separator=", "),
            root_le,
            root_te,
        )

        if getattr(self, "spanwise_enabled", False):
            tip_upper = np.array(self.upper_params_tip.detach().cpu().numpy(), dtype=float)
            tip_lower = np.array(self.lower_params_tip.detach().cpu().numpy(), dtype=float)
            tip_le = float(self.leading_edge_param_tip.detach().cpu().numpy()[0])
            tip_te = float(self.TE_thickness_param_tip.detach().cpu().numpy()[0])
            self.logger.info(
                "Iter %d | airfoil tip  | upper=%s | lower=%s | LE=%.6f | TE=%.6f",
                self._iter,
                np.array2string(tip_upper, precision=5, separator=", "),
                np.array2string(tip_lower, precision=5, separator=", "),
                tip_le,
                tip_te,
            )

    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        self._apply_gradients(upstream_grads)
        self.optimizer.step()
        self._step_scheduler(upstream_grads)
        self._enforce_constraints()
        
        if self._iter == self.config.run.max_outer_iters - 1 and not self.config.io.wandb.enabled:
            self.save_gif(fps=self.config.io.gif_fps)
        
        return {}

    def get_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return float(getattr(self.config.airfoil, "lr", 0.0))

    def _create_scheduler(self, af_conf):
        sched_conf = getattr(af_conf, "lr_schedule", None)
        if sched_conf is None:
            return None

        def _get(c, k, default):
            if c is None:
                return default
            if isinstance(c, dict):
                return c.get(k, default)
            return getattr(c, k, default)

        typ = _get(sched_conf, "type", "exponential")
        
        if typ == "exponential":
            gamma = _get(sched_conf, "gamma", 0.99)
            return lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif typ == "step":
            step_size = _get(sched_conf, "step_size", 100)
            gamma = _get(sched_conf, "gamma", 0.1)
            return lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif typ == "cosine":
            T_max = _get(sched_conf, "T_max", 100)
            return lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif typ == "reduce_on_plateau":
            mode = _get(sched_conf, "mode", "min")
            factor = _get(sched_conf, "factor", 0.1)
            patience = _get(sched_conf, "patience", 10)
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience, verbose=True)
        
        return None

    def _log_params_to_wandb(self):
        metrics = {"airfoil/learning_rate": self.get_lr()}
        
        for i, val in enumerate(self.upper_params.detach().numpy()):
            metrics[f"airfoil/upper_params_{i}"] = float(val)
        for i, val in enumerate(self.lower_params.detach().numpy()):
            metrics[f"airfoil/lower_params_{i}"] = float(val)
        
        metrics["airfoil/leading_edge_weight"] = float(self.leading_edge_param.detach().numpy()[0])
        metrics["airfoil/TE_thickness"] = float(self.TE_thickness_param.detach().numpy()[0])
        
        if getattr(self, "spanwise_enabled", False):
            for i, val in enumerate(self.upper_params_tip.detach().numpy()):
                metrics[f"airfoil_tip/upper_params_{i}"] = float(val)
            for i, val in enumerate(self.lower_params_tip.detach().numpy()):
                metrics[f"airfoil_tip/lower_params_{i}"] = float(val)

            metrics["airfoil_tip/leading_edge_weight"] = float(self.leading_edge_param_tip.detach().numpy()[0])
            metrics["airfoil_tip/TE_thickness"] = float(self.TE_thickness_param_tip.detach().numpy()[0])

        wandb.log(metrics, step=self._iter)

    def _apply_gradients(self, upstream_grads):
        self.optimizer.zero_grad()
        self.upper_params.grad = upstream_grads["dupper_params"]
        self.lower_params.grad = upstream_grads["dlower_params"]
        self.leading_edge_param.grad = upstream_grads["dleading_edge_param"]
        self.TE_thickness_param.grad = upstream_grads["dTE_thickness_param"]

        if getattr(self, "spanwise_enabled", False):
            self.upper_params_tip.grad = upstream_grads.get("dupper_params_tip", torch.zeros_like(self.upper_params_tip))
            self.lower_params_tip.grad = upstream_grads.get("dlower_params_tip", torch.zeros_like(self.lower_params_tip))
            self.leading_edge_param_tip.grad = upstream_grads.get("dleading_edge_param_tip", torch.zeros_like(self.leading_edge_param_tip))
            self.TE_thickness_param_tip.grad = upstream_grads.get("dTE_thickness_param_tip", torch.zeros_like(self.TE_thickness_param_tip))

            if getattr(self.config.io, "log_airfoil_coeffs", True):
                g_ru = self.upper_params.grad.detach()
                g_tu = self.upper_params_tip.grad.detach()
                g_rl = self.lower_params.grad.detach()
                g_tl = self.lower_params_tip.grad.detach()
                g_rle = self.leading_edge_param.grad.detach()
                g_tle = self.leading_edge_param_tip.grad.detach()
                g_rte = self.TE_thickness_param.grad.detach()
                g_tte = self.TE_thickness_param_tip.grad.detach()
                self.logger.info(
                    "Iter %d | grad root-tip Δ | upper=%.3e lower=%.3e LE=%.3e TE=%.3e",
                    self._iter,
                    float(torch.norm(g_ru - g_tu).cpu().item()),
                    float(torch.norm(g_rl - g_tl).cpu().item()),
                    float(torch.norm(g_rle - g_tle).cpu().item()),
                    float(torch.norm(g_rte - g_tte).cpu().item()),
                )

    def _step_scheduler(self, upstream_grads):
        if self.scheduler is None:
            return

        try:
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                metric = None
                if isinstance(upstream_grads, dict):
                    metric = upstream_grads.get("outer_loss", upstream_grads.get("loss", None))
                if metric is not None:
                    self.scheduler.step(metric)
                else:
                    warnings.warn("ReduceLROnPlateau scheduler configured but no metric found in upstream_grads; skipping step.")
            else:
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

            if getattr(self, "spanwise_enabled", False):
                self.TE_thickness_param_tip.clamp_(1e-4, 0.01)
                self.upper_params_tip.data = torch.maximum(
                    self.upper_params_tip.data,
                    self.lower_params_tip.data + min_gap
                )

    def plot(self):
        airfoilConfig = self.config.airfoil
        root_airfoil = asb.KulfanAirfoil(
            name=self.config.io.run_name + "_airfoil",
            lower_weights=self.lower_params.detach().numpy(),
            upper_weights=self.upper_params.detach().numpy(),
            leading_edge_weight=self.leading_edge_param.detach().numpy(),
            TE_thickness=self.TE_thickness_param.detach().numpy(),
            N1=airfoilConfig.N1,
            N2=airfoilConfig.N2,
        )

        tip_airfoil = None
        if getattr(self, "spanwise_enabled", False):
            tip_airfoil = asb.KulfanAirfoil(
                name=self.config.io.run_name + "_airfoil_tip",
                lower_weights=self.lower_params_tip.detach().numpy(),
                upper_weights=self.upper_params_tip.detach().numpy(),
                leading_edge_weight=self.leading_edge_param_tip.detach().numpy(),
                TE_thickness=self.TE_thickness_param_tip.detach().numpy(),
                N1=airfoilConfig.N1,
                N2=airfoilConfig.N2,
            )

        fig, ax = plt.subplots(figsize=(6, 3), dpi=200)

        x_root = np.reshape(np.array(root_airfoil.x()), -1)
        y_root = np.reshape(np.array(root_airfoil.y()), -1)

        ax.plot(x_root, y_root, ".-", color="#280887", zorder=11, label="Root")
        ax.fill(x_root, y_root, color="#280887", alpha=0.2, zorder=10)

        if tip_airfoil is not None:
            x_tip = np.reshape(np.array(tip_airfoil.x()), -1)
            y_tip = np.reshape(np.array(tip_airfoil.y()), -1)
            ax.plot(x_tip, y_tip, ".-", color="#2a9d8f", zorder=13, label="Tip")
            ax.fill(x_tip, y_tip, color="#2a9d8f", alpha=0.2, zorder=12)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        
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
