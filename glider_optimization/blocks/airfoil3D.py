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
from mpl_toolkits.mplot3d import Axes3D
import logging
from ..utils.spanwise_geometry import compute_dynamic_wing_reference_geometry, _section_centroid_from_kulfan

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
        
        tip_upper = af_conf.upper_initial_weights_tip if af_conf.upper_initial_weights_tip is not None else af_conf.upper_initial_weights
        tip_lower = af_conf.lower_initial_weights_tip if af_conf.lower_initial_weights_tip is not None else af_conf.lower_initial_weights
        tip_le = af_conf.leading_edge_weight_tip if af_conf.leading_edge_weight_tip is not None else af_conf.leading_edge_weight
        tip_te = af_conf.TE_thickness_tip if af_conf.TE_thickness_tip is not None else af_conf.TE_thickness

        self.upper_params_tip = nn.Parameter(torch.tensor(tip_upper, dtype=torch.float32))
        self.lower_params_tip = nn.Parameter(torch.tensor(tip_lower, dtype=torch.float32))
        self.leading_edge_param_tip = nn.Parameter(torch.tensor([tip_le], dtype=torch.float32))
        self.TE_thickness_param_tip = nn.Parameter(torch.tensor([tip_te], dtype=torch.float32))
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
        self._n_span_stations = 7

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self._iter = downstream_info["iteration"]

        plane_cfg = getattr(self.config, "plane", {}) or {}
        wing_cfg = plane_cfg.get("wing", {}) if isinstance(plane_cfg, dict) else {}
        dynamic_centroid = bool(wing_cfg.get("dynamic_centroid", False)) if isinstance(wing_cfg, dict) else False

        wing_reference_geometry = None
        if dynamic_centroid:
            root_kulfan = {
                "upper_weights": self.upper_params.detach().cpu().numpy(),
                "lower_weights": self.lower_params.detach().cpu().numpy(),
                "leading_edge_weight": float(self.leading_edge_param.detach().cpu().numpy()[0]),
                "TE_thickness": float(self.TE_thickness_param.detach().cpu().numpy()[0]),
            }
            tip_kulfan = {
                "upper_weights": self.upper_params_tip.detach().cpu().numpy(),
                "lower_weights": self.lower_params_tip.detach().cpu().numpy(),
                "leading_edge_weight": float(self.leading_edge_param_tip.detach().cpu().numpy()[0]),
                "TE_thickness": float(self.TE_thickness_param_tip.detach().cpu().numpy()[0]),
            }
            wing_reference_geometry = compute_dynamic_wing_reference_geometry(
                wing_cfg=wing_cfg,
                root_kulfan=root_kulfan,
                tip_kulfan=tip_kulfan,
                n_span_stations=self._n_span_stations,
            )

        if self._iter % self.config.io.log_every == 0:
            self.plot()
            if self.config.io.wandb.enabled:
                self._log_params_to_wandb()
                
        out = {
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
        if wing_reference_geometry is not None:
            out["wing_reference_geometry"] = wing_reference_geometry
        return out

    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        self._apply_gradients(upstream_grads)

        self.optimizer.step()
        self._step_scheduler()
        self._enforce_constraints()
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
        # Tip keys are absent in 2D mode (NeuralFoilSampling); fall back to root grads.
        self.upper_params_tip.grad = upstream_grads.get("dupper_params_tip", upstream_grads["dupper_params"])
        self.lower_params_tip.grad = upstream_grads.get("dlower_params_tip", upstream_grads["dlower_params"])
        self.leading_edge_param_tip.grad = upstream_grads.get("dleading_edge_param_tip", upstream_grads["dleading_edge_param"])
        self.TE_thickness_param_tip.grad = upstream_grads.get("dTE_thickness_param_tip", upstream_grads["dTE_thickness_param"])

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

        ax.plot(x_root, y_root, ".-", color="#280887", zorder=11)
        ax.fill(x_root, y_root, color="#280887", alpha=0.2, zorder=10)

        ax.plot(x_tip, y_tip, ".-", color="#d97706", zorder=13)
        ax.fill(x_tip, y_tip, color="#d97706", alpha=0.15, zorder=12)

        root_kulfan = {
            "upper_weights": self.upper_params.detach().numpy(),
            "lower_weights": self.lower_params.detach().numpy(),
            "leading_edge_weight": float(self.leading_edge_param.detach().numpy()[0]),
            "TE_thickness": float(self.TE_thickness_param.detach().numpy()[0]),
        }
        tip_kulfan = {
            "upper_weights": self.upper_params_tip.detach().numpy(),
            "lower_weights": self.lower_params_tip.detach().numpy(),
            "leading_edge_weight": float(self.leading_edge_param_tip.detach().numpy()[0]),
            "TE_thickness": float(self.TE_thickness_param_tip.detach().numpy()[0]),
        }

        cx_root, cz_root = _section_centroid_from_kulfan(root_kulfan)
        cx_tip, cz_tip = _section_centroid_from_kulfan(tip_kulfan)

        plane_cfg = getattr(self.config, "plane", {}) or {}
        wing_cfg = plane_cfg.get("wing", {}) if isinstance(plane_cfg, dict) else {}

        c_half_raw = np.array(wing_cfg.get("c_half", [1.0, 1.0]), dtype=float)
        y_half = np.array(wing_cfg.get("y_half", [0.0, 1.0]), dtype=float)

        if len(c_half_raw) != len(y_half):
            y_c = np.linspace(y_half[0], y_half[-1], len(c_half_raw))
            c_half = np.interp(y_half, y_c, c_half_raw)
        else:
            c_half = c_half_raw

        half_span = max(float(y_half[-1]), 1e-12)
        eta = np.clip(y_half / half_span, 0.0, 1.0)

        cx_span = (1.0 - eta) * cx_root + eta * cx_tip
        cz_span = (1.0 - eta) * cz_root + eta * cz_tip
        den = float(np.trapezoid(c_half, y_half))
        if den > 0:
            cx_wing = float(np.trapezoid(c_half * cx_span, y_half) / den)
            cz_wing = float(np.trapezoid(c_half * cz_span, y_half) / den)
        else:
            cx_wing, cz_wing = float(np.mean(cx_span)), float(np.mean(cz_span))

        ax.scatter(cx_root, cz_root, color="#280887", s=60, alpha=0.4)
        ax.scatter(cx_tip, cz_tip, color="#d97706", s=60, alpha=0.4)
        ax.scatter(cx_wing, cz_wing, color="black", s=60)

        ax.axis("off")
        ax.set_aspect(1.0, adjustable="datalim")

        fig.canvas.draw()
        buf, (w, h) = fig.canvas.print_to_buffer()
        frame_2d = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]

        try:
            frame_3d = self.plot_3d()
        except Exception:
            frame_3d = np.zeros_like(frame_2d)

        def to_rgb(img):
            if img.shape[2] == 4:
                return img[..., :3]
            return img

        frame_2d = to_rgb(frame_2d)
        frame_3d = to_rgb(frame_3d)

        h = max(frame_2d.shape[0], frame_3d.shape[0])

        def pad(img, h):
            if img.shape[0] == h:
                return img
            pad_h = h - img.shape[0]
            return np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant")

        frame_2d = pad(frame_2d, h)
        frame_3d = pad(frame_3d, h)

        combined = np.concatenate([frame_2d, frame_3d], axis=1)

        if self.config.io.wandb.enabled:
            wandb.log({"airfoil/shape": wandb.Image(combined)}, step=self._iter)
        else:
            self.frames.append(combined)

        plt.close(fig)

    def plot_3d(self):
        airfoilConfig = self.config.airfoil

        n_span = 30
        n_pts = 80

        span = np.linspace(0, 1, n_span)

        X, Y, Z = [], [], []

        for eta in span:
            upper = (1 - eta) * self.upper_params.detach().numpy() + eta * self.upper_params_tip.detach().numpy()
            lower = (1 - eta) * self.lower_params.detach().numpy() + eta * self.lower_params_tip.detach().numpy()
            le = (1 - eta) * self.leading_edge_param.item() + eta * self.leading_edge_param_tip.item()
            te = (1 - eta) * self.TE_thickness_param.item() + eta * self.TE_thickness_param_tip.item()

            af = asb.KulfanAirfoil(
                lower_weights=lower,
                upper_weights=upper,
                leading_edge_weight=le,
                TE_thickness=te,
                N1=airfoilConfig.N1,
                N2=airfoilConfig.N2,
            )

            af = af.repanel(n_points_per_side=n_pts // 2)

            x = np.array(af.x())
            z = np.array(af.y())

            chord = 1.0 - 0.3 * eta
            sweep = 0.25
            dihedral = 0.15

            x = x * chord + sweep * eta
            z = z * chord + dihedral * eta
            y = np.ones_like(x) * eta

            X.append(x)
            Y.append(y)
            Z.append(z)

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        fig = plt.figure(figsize=(6, 4), dpi=180)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            X, Y, Z,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.95
        )

        ax.set_axis_off()

        ax.view_init(elev=18, azim=-60)

        max_range = np.array([
            X.max() - X.min(),
            Y.max() - Y.min(),
            Z.max() - Z.min()
        ]).max()

        mid_x = (X.max() + X.min()) / 2
        mid_y = (Y.max() + Y.min()) / 2
        mid_z = (Z.max() + Z.min()) / 2

        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        fig.canvas.draw()
        buf, (w, h) = fig.canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]

        plt.close(fig)
        return frame


    def save_gif(self, filename="airfoil_evolution.gif", fps=1):        
        if self.frames:
            log_dir = Path(self.config.io.checkpoint_dir)
            imageio.mimsave(log_dir/filename, self.frames, fps=fps)
