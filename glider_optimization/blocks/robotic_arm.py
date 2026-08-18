from ..blockBase import Block
from typing import override
from ..config import Config
from ..utils.arm_jinenv import ArmThrowing
from pathlib import Path
from typing import Dict, Any
from casadi import SX, Function, jacobian, vertcat
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

JOINT_NAMES = ["sh-az", "sh-el", "elbow", "wrist"]
SEGMENT_NAMES = ["upper arm", "forearm", "hand"]
JOINT_COLORS = ["#E63946", "#06A77D", "#118AB2", "#9D4EDD"]
SEGMENT_COLORS = ["#2E86AB", "#A23B72", "#F77F00"]


class RoboticArm(Block):
    """The design block of the throwing-arm co-design problem.

    Holds psi (3 segment lengths, 3 tapers, 3 structural-mass logits, 4 torque
    logits) and takes the Adam step. The OCP differentiates with respect to its
    auxvar -- the inertial parameters and the torque capacities -- so the
    gradient that arrives here has to be pulled back through the design map
    psi -> (auxvar, caps) before it means anything about psi.
    """

    @override
    def __init__(self, config: Config):
        self.config = config
        arm_conf = self.config.arm

        self.env = ArmThrowing(config)
        self._build_design_jacobian()

        self.design_parameters = nn.Parameter(
            torch.tensor(arm_conf.initial_psi, dtype=torch.float32)
        )

        self.optimizer = torch.optim.Adam(
            [self.design_parameters],
            lr=arm_conf.lr
        )

        self._iter = 0
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=arm_conf.gamma)
        self.frames = []

    def _build_design_jacobian(self):
        """d(auxvar, caps)/d(psi): the chain rule between OCP and design space."""
        psi = SX.sym("psi", self.env.n_psi)
        aux = vertcat(self.env.psi_to_auxvar(psi), self.env.psi_to_caps(psi))
        self.daux_dpsi_fn = Function("daux_dpsi", [psi], [jacobian(aux, psi)])

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self._iter = downstream_info["iteration"]

        if self._iter % self.config.io.log_every == 0:
            self.plot()
            if self.config.io.wandb.enabled:
                self._log_params_to_wandb()

        return {
            "params": self.design_parameters,
            "iteration": downstream_info["iteration"],
            # The budgets are enforced exactly by the softmax in the design map
            # and the box constraints by clipping, so there is no penalty term
            # to carry: kept in the payload because the pipeline expects it.
            "augmented_lagrangian": 0.0,
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
        weights = [checkpoint[f"arm/psi_{i}"] for i in range(self.env.n_psi)]

        self.design_parameters = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

        self.optimizer = torch.optim.Adam(
            [self.design_parameters],
            lr=self.config.arm.lr
        )
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.arm.gamma)

    def get_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return float(getattr(self.config.arm, "lr", 0.0))

    def design_summary(self) -> Dict[str, np.ndarray]:
        """psi decoded into the quantities that actually describe the arm."""
        psi = self.design_parameters.detach().cpu().numpy().ravel()
        struct_mass = _softmax(psi[6:9]) * self.env.M_STRUCT
        caps = np.asarray(self.env.psi_to_caps(psi)).ravel()
        aux = np.asarray(self.env.psi_to_auxvar(psi)).ravel()
        ns = self.env.N_SEG
        return {
            "psi": psi,
            "lengths": psi[0:3],
            "tapers": psi[3:6],
            "struct_mass": struct_mass,
            "motor_mass": self.env.K_MOTOR * caps,
            "caps": caps,
            "no_load_speed": self.config.arm.p_joint / caps,
            "segment_mass": aux[ns:2*ns],
            "segment_com": aux[2*ns:3*ns],
            "total_mass": float(self.env.psi_to_mass(psi)),
            "aux": aux,
        }

    def _log_params_to_wandb(self):
        d = self.design_summary()
        metrics = {
            "arm/learning_rate": self.get_lr(),
            "arm/total_mass": d["total_mass"],
            "arm/reach": float(d["lengths"].sum()),
        }

        for i, val in enumerate(d["psi"]):
            metrics[f"arm/psi_{i}"] = float(val)
        for b, name in enumerate(SEGMENT_NAMES):
            metrics[f"arm/length_{name}"] = float(d["lengths"][b])
            metrics[f"arm/taper_{name}"] = float(d["tapers"][b])
            metrics[f"arm/struct_mass_{name}"] = float(d["struct_mass"][b])
            metrics[f"arm/segment_mass_{name}"] = float(d["segment_mass"][b])
        for j, name in enumerate(JOINT_NAMES):
            metrics[f"arm/torque_capacity_{name}"] = float(d["caps"][j])
            metrics[f"arm/no_load_speed_{name}"] = float(d["no_load_speed"][j])

        wandb.log(metrics, step=self._iter)

    def _apply_gradients(self, upstream_grads):
        self.optimizer.zero_grad()

        dJ_daux = upstream_grads["dJ_dphi"].detach().cpu().numpy().reshape(-1, 1)
        psi = self.design_parameters.detach().cpu().numpy().ravel()
        daux_dpsi = np.asarray(self.daux_dpsi_fn(psi))

        dJ_dpsi = daux_dpsi.T @ dJ_daux

        if not np.isfinite(dJ_dpsi).all():
            warnings.warn("Non-finite design gradient; skipping this update.")
            dJ_dpsi = np.zeros_like(dJ_dpsi)

        self.design_parameters.grad = torch.from_numpy(
            dJ_dpsi.ravel()
        ).float().to(self.design_parameters.device)

    def _step_scheduler(self):
        if self.scheduler is None:
            return

        try:
            self.scheduler.step()
        except Exception:
            warnings.warn("LR scheduler step failed; continuing without scheduling.")

    def _enforce_constraints(self):
        arm_conf = self.config.arm
        with torch.no_grad():
            # Lengths and tapers are geometry and have to stay physical; the two
            # logit groups need no clipping because the softmax in the design map
            # already holds the mass and torque budgets exactly.
            self.design_parameters.data[0:3].clamp_(arm_conf.length_min, arm_conf.length_max)
            self.design_parameters.data[3:6].clamp_(arm_conf.taper_min, arm_conf.taper_max)

    def plot(self):
        """The arm as designed: true-scale profile plus the two budget splits."""
        d = self.design_summary()

        fig = plt.figure(figsize=(11, 4.5), dpi=140)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], hspace=0.55, wspace=0.25)
        ax = fig.add_subplot(gs[:, 0])

        self._draw_arm(ax, d)

        ax_m = fig.add_subplot(gs[0, 1])
        ax_m.bar(SEGMENT_NAMES, d["struct_mass"], color=SEGMENT_COLORS, alpha=0.85)
        ax_m.set_ylabel("kg", fontsize=8)
        ax_m.set_title(f"structural mass  (budget {self.env.M_STRUCT:.2f} kg)",
                       fontsize=9, fontweight="bold")

        ax_c = fig.add_subplot(gs[1, 1])
        ax_c.bar(JOINT_NAMES, d["caps"], color=JOINT_COLORS, alpha=0.85)
        ax_c.set_ylabel("N m", fontsize=8)
        ax_c.set_title(f"torque capacity  (budget {self.env.C_TOTAL:.0f} N m)",
                       fontsize=9, fontweight="bold")

        for a in (ax_m, ax_c):
            a.tick_params(labelsize=8)
            a.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"Arm design - iteration {self._iter}   "
                     f"(reach {d['lengths'].sum():.3f} m, "
                     f"total mass {d['total_mass']:.2f} kg)",
                     fontsize=10, fontweight="bold")

        fig.canvas.draw()
        buf, (w, h) = fig.canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]

        if self.config.io.wandb.enabled:
            wandb.log({"arm/design": wandb.Image(frame, caption=f"Arm Iter {self._iter}")},
                      step=self._iter)
        else:
            self.frames.append(frame)

        plt.close(fig)

    def _draw_arm(self, ax, d):
        """Side view of the arm in its wind-up pose, drawn to scale."""
        lengths = d["lengths"]
        tapers = d["tapers"]
        radii = np.asarray(self.env.R0, dtype=float)
        q = np.asarray(self.config.arm.q_start, dtype=float)

        # planar (x, z) pose: azimuth q[0] only turns the whole arm about z
        origin = np.array([0.0, self.env.shoulder_h])
        angle = 0.0
        p = origin
        for b in range(self.env.N_SEG):
            angle = angle + q[b + 1]
            direction = np.array([np.cos(angle), -np.sin(angle)])
            normal = np.array([-direction[1], direction[0]])

            s = np.linspace(0.0, 1.0, 24)
            centres = p[None, :] + s[:, None] * lengths[b] * direction[None, :]
            r = radii[b] * (1.0 + (tapers[b] - 1.0) * s)
            upper = centres + r[:, None] * normal[None, :]
            lower = centres - r[:, None] * normal[None, :]
            outline = np.vstack([upper, lower[::-1]])

            ax.fill(outline[:, 0], outline[:, 1], color=SEGMENT_COLORS[b],
                    alpha=0.55, zorder=3)
            ax.plot(outline[:, 0], outline[:, 1], color=SEGMENT_COLORS[b],
                    lw=1.2, zorder=4)

            # motors sit at the proximal end of the segment they drive, drawn at
            # the size their share of the torque budget pays for
            seg_caps = sum(d["caps"][j] for j, seg in enumerate(self.env.JOINT_SEG)
                           if seg == b)
            motor_r = 0.012 + 0.05 * seg_caps / self.env.C_TOTAL
            ax.add_patch(plt.Circle(p, motor_r, color="#333333", alpha=0.8, zorder=5))

            p = p + lengths[b] * direction

        ax.plot([0, 0], [0, self.env.shoulder_h], color="#555", lw=3, alpha=0.7, zorder=1)
        ax.plot(*p, "o", color="#E63946", markersize=7, zorder=6)   # the ball

        span = float(np.sum(lengths)) + 0.15
        ax.set_xlim(origin[0] - span, origin[0] + span)
        ax.set_ylim(0.0, self.env.shoulder_h + span)
        ax.set_aspect(1.0)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("z (m)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_title("wind-up pose, to scale", fontsize=9, fontweight="bold")

    def save_gif(self, filename="arm_evolution.gif", fps=1):
        if self.frames:
            log_dir = Path(self.config.io.checkpoint_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(log_dir/filename, self.frames, fps=fps)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
