from pathlib import Path
from ..blockBase import Block
#from typing import Dict, Any, override, List, Optional
from typing_extensions import Dict, Any, override, List, Optional
from ..utils.go_safe_pdp import COCsys
from ..utils.glider_jinenv import GliderPerching
from ..utils.idoc_ineq import build_blocks_idoc, idoc_full
from ..config import Config, EvaluationMode
from casadi import pi, vertcat, DM, Function
import csv
import numpy as np
import torch
import logging
import wandb
import tempfile
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from math import sqrt

def solve_worker(config: Config, 
                 init_state: List[float], 
                 auxvar_vector: np.ndarray, 
                 wing_reference_geometry: Optional[Dict[str, float]] = None,
                 prev_w_opt: Optional[List[float]] = None, 
                 prev_lam_g: Optional[List[float]] = None, 
                 prev_lam_x: Optional[List[float]] = None) -> Dict[str, Any]:

    env = GliderPerching(config, wing_reference_geometry=wing_reference_geometry)
    coc = COCsys()
    
    env.initDyn()
    f_fun = Function(
        "f_fun",
        [env.X, env.U, env.dyn_auxvar],
        [env.f]
    )

    X = env.X
    U = env.U
    P = env.dyn_auxvar
    dt = X[-1]

    k1 = f_fun(X, U, P)
    k2 = f_fun(X + 0.5*dt*k1, U, P)
    k3 = f_fun(X + 0.5*dt*k2, U, P)
    k4 = f_fun(X + dt*k3, U, P)

    X_next = X + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    env.initCost(state_weights=config.ocp.terminal_state_weight, wu=config.ocp.stage_control_weight, init_state=init_state)
    env.initConstraints(-pi/3, pi/8, 13)
    
    coc.setAuxvarVariable(vertcat(env.dyn_auxvar))
    coc.setStateVariable(env.X)
    coc.setControlVariable(env.U)
    coc.setDyn(X_next)

    coc.setPathCost(env.path_cost)
    coc.setFinalCost(env.final_cost)

    coc.setPathInequCstr(env.path_inequ)
    #coc.setFinalInequCstr(env.final_inequ)
    
    coc.diffCPMP()
    
    warm_start = False
    if prev_w_opt is not None and prev_lam_g is not None:
        warm_start = True
        coc.w_opt_prev = prev_w_opt
        coc.lam_g_prev = prev_lam_g
        coc.lam_x_prev = prev_lam_x

    res = coc.ocSolver(horizon=111, 
                       init_state=init_state, 
                       auxvar_value=auxvar_vector, 
                       timeVarying=True, 
                       warm_start=warm_start)
    
    # This shouldn't be part of the forward, however CasaDi functions cannot be pickled.   
    try:
        auxsys_COC = coc.getAuxSys(opt_sol=res, threshold=1e-5)
        res['auxsys_COC'] = auxsys_COC
    except Exception as e:
        res['auxsys_COC'] = None
        print(f"Warning: Failed to extract sensitivity data: {e}")
        
    return res

class OCP(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.run.device)
        self.logger = logging
        
        # Not used, kept only for plotting/animation purposes
        self.env = GliderPerching(self.config)
        self.env.state_weights = config.ocp.terminal_state_weight
        self.last_trajs: List[Dict[str, Any]] = [] 
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        wing_reference_geometry = downstream_info.get("wing_reference_geometry")
        self._wing_reference_geometry = wing_reference_geometry
        
        weights_CL = downstream_info["phi_CL"].view(-1, 1).detach().cpu().numpy()
        weights_CD = downstream_info["phi_CD"].view(-1, 1).detach().cpu().numpy()
        weights_CM = downstream_info["phi_CM"].view(-1, 1).detach().cpu().numpy()
        
        auxvar_vector = np.vstack([weights_CL, weights_CD, weights_CM])
        self._auxvar_vector = auxvar_vector  # stored for CSV wing-force evaluation
        
        initial_states = self._get_initial_states_for_mode()
        num_states = len(initial_states)
        
        worker_args = []
        for i, init_state in enumerate(initial_states):
            prev_w, prev_lg, prev_lx = None, None, None
            if len(self.last_trajs) > i:
                 prev_sol = self.last_trajs[i]
                 if prev_sol["success"]:
                    prev_w = prev_sol.get("w_opt")
                    prev_lg = prev_sol.get("lam_g")
                    prev_lx = prev_sol.get("lam_x")
            
            worker_args.append((self.config, init_state, auxvar_vector, wing_reference_geometry, prev_w, prev_lg, prev_lx))

        num_workers = min(mp.cpu_count(), len(worker_args))
        if num_workers < 1: num_workers = 1
        
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(solve_worker, worker_args)
            
        self.last_trajs = results
        
        failures = sum(1 for r in results if not r["success"])
        if failures > 0:
            self.logger.warning(f"⚠️ {failures}/{num_states} IPOPT solves failed")

        self._log_flight_conditions(downstream_info["iteration"], wing_reference_geometry)
            
        num_iterations = self.config.run.max_outer_iters
        iteration = downstream_info["iteration"]
        self._it = iteration
        static_every = max(1, int(getattr(self.config.io, "static_plot_every", self.config.io.log_every)))
        
        if iteration % static_every == 0 or iteration == (num_iterations - 1):
            self.plot_static(iteration)
            
        #if iteration == 0 or iteration == (num_iterations - 1):
        #    self.plot_animations(iteration)

        end_time = time.perf_counter()
        forward_time = end_time - start_time
        
        if self.config.io.wandb.enabled:
            wandb.log({
                "profiler/ocp_forward_time": forward_time,
            }, step=iteration)

        return {
            "trajectory": self.last_trajs, 
            "iteration": downstream_info["iteration"],
            "augmented_lagrangian": downstream_info["augmented_lagrangian"] 
        }

    def _get_initial_states_for_mode(self) -> List[List[float]]:
        mode = self.config.evaluation.mode
        ocp_cfg = self.config.ocp

        if mode == EvaluationMode.SoftLanding:
            states = getattr(ocp_cfg, "initial_states_softlanding", None)
        else:
            states = getattr(ocp_cfg, "initial_states_perching", None)

        if states is None or len(states) == 0:
            states = ocp_cfg.initial_states

        if states is None or len(states) == 0:
            raise RuntimeError(f"No initial states configured for mode={mode}")

        return states

    def _chebyshev_nodes(self, a: float, b: float, n: int) -> np.ndarray:
        k = np.arange(n)
        return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) / (2 * n) * np.pi)

    def _log_flight_conditions(self, iteration: int, wing_reference_geometry: Optional[Dict[str, float]] = None) -> None:
        out_dir = Path(self.config.io.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "flight_conditions.log"

        if not out_file.exists():
            with out_file.open("w", encoding="utf-8") as f:
                f.write("flight_conditions.log\n")
                f.write("Per-iteration flight-condition diagnostics (file-only).\n")
                f.write("- alpha_grid_deg: Chebyshev AoA axis used to build AoA-Re mesh\n")
                f.write("- re_grid: Chebyshev Reynolds axis used to build AoA-Re mesh\n")
                f.write("- velocity_*: [vx, vz, |v|] from wing reference-point velocity\n")
                f.write("- alpha_*_deg: alpha = theta - atan2(vz, vx) in degrees\n")
                f.write("- at(traj, idx, x, z): trajectory id, point index, and state position\n")

        nf = self.config.neuralFoilSampling
        n_1d = int(sqrt(nf.n_samples))
        aoa_grid = self._chebyshev_nodes(float(nf.AoA_min), float(nf.AoA_max), n_1d) if n_1d > 0 else np.array([])
        re_grid = self._chebyshev_nodes(float(nf.Re_min), float(nf.Re_max), n_1d) if n_1d > 0 else np.array([])

        l_w_i = -0.005
        l_w_f = -0.015
        l_w_m = 0.5 * (l_w_i + l_w_f)
        if isinstance(wing_reference_geometry, dict):
            try:
                l_w_m = float(wing_reference_geometry.get("l_w_m", l_w_m))
            except Exception:
                pass

        points = []
        for traj_idx, traj in enumerate(self.last_trajs):
            states = traj.get("state_traj_opt")
            if states is None:
                continue
            states = np.asarray(states)
            if states.ndim != 2 or states.shape[1] < 8 or states.shape[0] == 0:
                continue

            x = states[:, 0]
            z = states[:, 1]
            theta = states[:, 2]
            xdot = states[:, 4]
            zdot = states[:, 5]
            thetadot = states[:, 6]

            vx = xdot + l_w_m * thetadot * np.sin(theta)
            vz = zdot - l_w_m * thetadot * np.cos(theta)
            speed = np.sqrt(vx * vx + vz * vz)
            alpha_deg = np.rad2deg(theta - np.arctan2(vz, vx))

            for i in range(states.shape[0]):
                points.append((traj_idx, i, x[i], z[i], vx[i], vz[i], speed[i], alpha_deg[i]))

        with out_file.open("a", encoding="utf-8") as f:
            f.write("=" * 88 + "\n")
            f.write(f"iteration={iteration}\n")
            f.write(f"grid_n_samples={int(nf.n_samples)} n_1d={n_1d} grid_points={n_1d * n_1d}\n")

            if aoa_grid.size > 0:
                aoa_text = ", ".join(f"{v:.6f}" for v in aoa_grid.tolist())
                f.write(f"alpha_grid_deg=[{aoa_text}]\n")
            else:
                f.write("alpha_grid_deg=[]\n")

            if re_grid.size > 0:
                re_text = ", ".join(f"{v:.6f}" for v in re_grid.tolist())
                f.write(f"re_grid=[{re_text}]\n")
            else:
                f.write("re_grid=[]\n")

            if not points:
                f.write("trajectory_stats=unavailable (no state_traj_opt data)\n")
                return

            speeds = np.array([p[6] for p in points], dtype=float)
            alphas = np.array([p[7] for p in points], dtype=float)

            p_vmin = points[int(np.argmin(speeds))]
            p_vmax = points[int(np.argmax(speeds))]
            p_amin = points[int(np.argmin(alphas))]
            p_amax = points[int(np.argmax(alphas))]

            f.write(
                "velocity_min[vx,vz,|v|]="
                f"[{p_vmin[4]:.6f}, {p_vmin[5]:.6f}, {p_vmin[6]:.6f}] "
                f"at(traj={p_vmin[0]}, idx={p_vmin[1]}, x={p_vmin[2]:.6f}, z={p_vmin[3]:.6f})\n"
            )
            f.write(
                "velocity_max[vx,vz,|v|]="
                f"[{p_vmax[4]:.6f}, {p_vmax[5]:.6f}, {p_vmax[6]:.6f}] "
                f"at(traj={p_vmax[0]}, idx={p_vmax[1]}, x={p_vmax[2]:.6f}, z={p_vmax[3]:.6f})\n"
            )
            f.write(
                f"alpha_min_deg={p_amin[7]:.6f} "
                f"at(traj={p_amin[0]}, idx={p_amin[1]}, x={p_amin[2]:.6f}, z={p_amin[3]:.6f})\n"
            )
            f.write(
                f"alpha_max_deg={p_amax[7]:.6f} "
                f"at(traj={p_amax[0]}, idx={p_amax[1]}, x={p_amax[2]:.6f}, z={p_amax[3]:.6f})\n"
            )
    
    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        delta = 0.00001
        dJ_deps_list = upstream_grads["dJ_deps"]
        
        total_dJ_dphi_np = None
        n_valid = 0

        for i, traj in enumerate(self.last_trajs):
            auxsys_COC = traj.get('auxsys_COC')
            if auxsys_COC is None or not traj.get('success', False):
                self.logger.warning(f"Skipping trajectory {i} in backward (IPOPT failed or no sensitivity data)")
                continue

            dJ_deps = dJ_deps_list[i]
            idoc_ctx = build_blocks_idoc(auxsys_COC, delta)
            traj_deriv_COC = idoc_full(idoc_ctx)
            
            deps_dphi = traj_deriv_COC['state_traj_opt']
            dJ_dphi_partial = np.einsum('ij,ijk->k', dJ_deps, deps_dphi).reshape(2028,1)
            
            if total_dJ_dphi_np is None:
                total_dJ_dphi_np = dJ_dphi_partial
            else:
                total_dJ_dphi_np += dJ_dphi_partial
            n_valid += 1

        if n_valid == 0:
            self.logger.error("All trajectories failed — returning zero gradient")
            total_dJ_dphi_np = np.zeros((2028, 1))
            n_valid = 1

        total_dJ_dphi_np /= n_valid

        dJ_dphi = torch.from_numpy(total_dJ_dphi_np).float().to(self.device)
        dJ_dphi = dJ_dphi.view(3, -1)

        end_time = time.perf_counter()
        backward_time = end_time - start_time
                
        if self.config.io.wandb.enabled:
            wandb.log({
                "profiler/ocp_backward_time": backward_time,
            }, step=self._it)

        return {"dJ_dphi": dJ_dphi}
                
    def plot_static(self, iteration):
        fig, ax = self._build_static_figure(iteration, f"All Trajectories - Iteration {iteration}")

        if self.config.io.wandb.enabled:
            wandb.log(
                {f"trajectory/all_trajectories_static": wandb.Image(fig)},
                step=iteration
            )
        else:
            run_name = getattr(self.config.io, "run_name", "run")
            out_dir = Path(self.config.io.checkpoint_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            static_plot_path = out_dir / f"{run_name}_static_iter{iteration}.png"
            fig.savefig(static_plot_path)

        plt.close(fig)

    def save_best_snapshot(self, metric_name: str, metric_value: float, best_iteration: int, filename_suffix: str) -> None:
        run_name = getattr(self.config.io, "run_name", "run")
        out_dir = Path(self.config.io.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        title = f"Best {metric_name} Trajectories"
        fig, ax = self._build_static_figure(best_iteration, title)

        overlay_text = f"{metric_name}: {metric_value:.6f}\nIteration: {best_iteration}"
        ax.text(
            0.02,
            0.98,
            overlay_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "black"},
        )

        filepath = out_dir / f"{run_name}_{filename_suffix}.png"
        fig.savefig(filepath)
        plt.close(fig)
        self.logger.info(f"Saved best {metric_name.lower()} snapshot to {filepath}")

        self._save_trajectory_csvs(out_dir, run_name, filename_suffix)

    def _save_trajectory_csvs(self, out_dir: Path, run_name: str, suffix: str) -> None:
        """Write one CSV per trajectory.

        All forces and alpha_w are evaluated directly from the CasADi dynamics
        surrogate (debug_f), which is exact and consistent with what IPOPT optimised.

        Columns:
          step, x_m, z_m, vx_ms, vz_ms,
          theta_deg            – body pitch angle from horizontal (= wing orientation)
          alpha_w_deg          – wing AoA: theta minus flight-path angle at c/4
          Fx_N, Fz_N           – total plane aero force (wing + elevator), global frame
          Fx_wing_N, Fz_wing_N – wing-only aerodynamic force, global frame

        All velocities and forces are in the global inertial (world) frame.
        alpha_w_deg uses l_w_ac (c/4) as the evaluation arm — identical to the dynamics.
        The last row has NaN for forces/alpha (no control available at terminal step).
        """
        wing_ref = getattr(self, "_wing_reference_geometry", None)
        auxvar_vector = getattr(self, "_auxvar_vector", None)

        # Compute l_w_ac (c/4 arm) from wing planform — matches glider_jinenv.py exactly.
        # Used as fallback for alpha_w when debug_fn result is unavailable.
        plane_cfg = self.config.plane if isinstance(self.config.plane, dict) else {}
        dyn_cfg   = plane_cfg.get("dyn", {}) if isinstance(plane_cfg, dict) else {}
        wing_cfg  = plane_cfg.get("wing", {}) if isinstance(plane_cfg, dict) else {}
        l_w_i   = float(dyn_cfg.get("l_w_i", -0.005))
        l_w_f   = float(dyn_cfg.get("l_w_f", -0.015))
        l_w_ac  = l_w_i + 0.25 * (l_w_f - l_w_i)   # default: c/4
        y_half  = wing_cfg.get("y_half") if isinstance(wing_cfg, dict) else None
        c_half  = wing_cfg.get("c_half") if isinstance(wing_cfg, dict) else None
        xle_half = wing_cfg.get("xle_half") if isinstance(wing_cfg, dict) else None
        if (isinstance(y_half, list) and isinstance(c_half, list)
                and isinstance(xle_half, list) and len(y_half) >= 2):
            y  = np.asarray(y_half, dtype=float)
            c  = np.asarray(c_half, dtype=float)
            xle = np.asarray(xle_half, dtype=float)
            den = float(np.trapezoid(c, y))
            if den > 0:
                l_w_ac = float(np.trapezoid(c * (xle + 0.25 * c), y) / den)

        # Build CasADi debug_f once for all trajectories.
        # debug_f(X, U, auxvar) -> [alpha_w, v_w, CL_w, CD_w, CM_w, F_w[2], F_e[2], ...]
        debug_fn = None
        if auxvar_vector is not None:
            try:
                _env = GliderPerching(self.config, wing_reference_geometry=wing_ref)
                _env.initDyn()
                debug_fn = _env.debug_f
            except Exception as e:
                self.logger.warning(f"Could not build debug_f for CSV forces: {e}")

        for i, traj in enumerate(self.last_trajs):
            if not traj.get("success", False):
                continue
            states = traj.get("state_traj_opt")
            if states is None:
                continue
            states = np.asarray(states)
            if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 8:
                continue

            controls_raw = traj.get("control_traj_opt")
            controls = np.asarray(controls_raw) if controls_raw is not None else None

            N = states.shape[0]
            x        = states[:, 0]
            z        = states[:, 1]
            theta    = states[:, 2]
            vx       = states[:, 4]
            vz       = states[:, 5]
            thetadot = states[:, 6]
            theta_deg = np.rad2deg(theta)

            # Fallback alpha_w: evaluated at c/4 (l_w_ac), matching the dynamics
            vx_w = vx + l_w_ac * thetadot * np.sin(theta)
            vz_w = vz - l_w_ac * thetadot * np.cos(theta)
            alpha_w_deg = np.rad2deg(theta - np.arctan2(vz_w, vx_w))

            # Forces and alpha_w from debug_f (exact surrogate, global frame)
            Fx_wing = np.full(N, np.nan)
            Fz_wing = np.full(N, np.nan)
            Fx      = np.full(N, np.nan)  # total = wing + elevator
            Fz      = np.full(N, np.nan)
            if debug_fn is not None and controls is not None and auxvar_vector is not None:
                n_ctrl = controls.shape[0]
                for k in range(min(N - 1, n_ctrl)):
                    try:
                        u_k = float(np.asarray(controls[k]).reshape(-1)[0])
                        out = debug_fn(states[k], [u_k], auxvar_vector)
                        # outputs: alpha_w, v_w, CL_w, CD_w, CM_w, F_w[2], F_e[2], ...
                        alpha_w_deg[k] = float(np.rad2deg(float(np.asarray(out[0]).reshape(-1)[0])))
                        F_w = np.asarray(out[5]).reshape(-1)
                        F_e = np.asarray(out[6]).reshape(-1)
                        Fx_wing[k] = float(F_w[0])
                        Fz_wing[k] = float(F_w[1])
                        Fx[k]      = float(F_w[0] + F_e[0])
                        Fz[k]      = float(F_w[1] + F_e[1])
                    except Exception:
                        pass  # leave fallback alpha_w and NaN forces

            def _fmt(v):
                return "" if np.isnan(v) else f"{v:.6f}"

            filepath = out_dir / f"{run_name}_{suffix}_traj{i}.csv"
            with filepath.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "x_m", "z_m", "vx_ms", "vz_ms",
                                 "theta_deg", "alpha_w_deg",
                                 "Fx_N", "Fz_N", "Fx_wing_N", "Fz_wing_N"])
                for k in range(N):
                    writer.writerow([
                        k,
                        f"{x[k]:.6f}",
                        f"{z[k]:.6f}",
                        f"{vx[k]:.6f}",
                        f"{vz[k]:.6f}",
                        f"{theta_deg[k]:.6f}",
                        f"{alpha_w_deg[k]:.6f}",
                        _fmt(Fx[k]),
                        _fmt(Fz[k]),
                        _fmt(Fx_wing[k]),
                        _fmt(Fz_wing[k]),
                    ])
            self.logger.info(f"Saved trajectory CSV ({N} steps) to {filepath}")

    def _build_static_figure(self, iteration: int, title: str):
        run_name = getattr(self.config.io, "run_name", "run")

        fig, ax = plt.subplots(figsize=(10, 8))

        num_trajs = len(self.last_trajs)
        colors = plt.cm.viridis(np.linspace(0, 1, num_trajs))

        for i, traj in enumerate(self.last_trajs):
            states = traj.get('state_traj_opt')
            if states is None:
                continue
            states = np.asarray(states)
            if states.ndim != 2 or states.shape[0] == 0:
                continue

            ax.plot(states[:, 0], states[:, 1], color=colors[i], alpha=0.7, linewidth=1.5, zorder=1)

            ax.scatter(states[0, 0], states[0, 1], color=colors[i], marker='o', s=40, zorder=2)

        ax.scatter(0, 0, color='red', marker='x', s=100, linewidth=3, zorder=3)

        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, label='Trajectories'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Start'),
            Line2D([0], [0], marker='x', color='red', markersize=10, markeredgewidth=2, label='Target')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(title)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Z Position (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        return fig, ax

    def plot_animations(self, iteration):
        run_name = getattr(self.config.io, "run_name", "run")
        samples_to_plot = self.last_trajs
        
        for i, traj in enumerate(samples_to_plot):
            suffix = f"_ic{i}"
            if self.config.io.wandb.enabled:
                with tempfile.TemporaryDirectory() as tmpdir:
                    title = Path(tmpdir) / f"{run_name}_traj_iter{iteration}{suffix}"
                    self.env.play_animation(
                        traj['state_traj_opt'],
                        traj['control_traj_opt'],
                        save_option=True,
                        title=str(title),
                        fps=self.config.io.gif_fps,
                    )
                    gif_path = f"{title}.gif"
                    wandb.log(
                        {f"trajectory/traj_iter_{iteration}_ic{i}": wandb.Video(gif_path, format="gif")},
                        step=iteration
                    )
            else:
                out_dir = Path(self.config.io.checkpoint_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                title = out_dir / f"{run_name}_traj_iter{iteration}{suffix}"
                self.env.play_animation(
                    traj['state_traj_opt'],
                    traj['control_traj_opt'],
                    save_option=True,
                    title=str(title),
                    fps=self.config.io.gif_fps
                )