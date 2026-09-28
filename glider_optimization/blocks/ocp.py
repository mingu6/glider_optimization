from pathlib import Path
from ..blockBase import Block
from typing import Dict, Any, override, List, Optional
from ..utils.go_safe_pdp import COCsys
from ..utils.glider_jinenv import GliderPerching
from ..utils.idoc_ineq import build_blocks_idoc, idoc_full
from ..config import Config, EvaluationMode
from casadi import pi, vertcat, DM, Function
import os
import pickle

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

def solve_worker(config: Config,
                 task_mode: Optional[str],
                 task_horizon: Optional[int],
                 task_wu: Optional[float],
                 init_state: List[float],
                 auxvar_vector: np.ndarray,
                 wing_geometry: Optional[Dict[str, Any]] = None,
                 prev_w_opt: Optional[List[float]] = None,
                 prev_lam_g: Optional[List[float]] = None,
                 prev_lam_x: Optional[List[float]] = None) -> Dict[str, Any]:

    if task_mode is not None:
        # Each task carries its own OCP; the worker gets its own pickled copy of the
        # config, so retargeting the mode here is local to this solve.
        config = config.model_copy(deep=True)
        config.evaluation.mode = EvaluationMode(task_mode)

    env = GliderPerching(config, wing_geometry=wing_geometry)
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

    env.initCost(state_weights=config.ocp.terminal_state_weight,
                 wu=config.ocp.stage_control_weight if task_wu is None else task_wu,
                 init_state=init_state)
    env.initConstraints(-pi/3, pi/8, 2.2)
    
    coc.setAuxvarVariable(vertcat(env.dyn_auxvar))
    coc.setStateVariable(env.X)
    coc.setControlVariable(env.U)
    coc.setDyn(X_next)
    # z is state 1; keeps the NLP's initial guess from starting underground.
    coc.ground_index = 1

    coc.setPathCost(env.path_cost)
    coc.setFinalCost(env.final_cost)

    coc.setPathInequCstr(env.path_inequ)
    # NOTE: terminal inequality constraints are not supported by the IDOC backward
    # pass (it sizes the terminal block from the state alone), so touchdown softness
    # is enforced through the terminal cost instead. See sink_weight.
    #coc.setFinalInequCstr(env.final_inequ)
    
    coc.diffCPMP()
    
    warm_start = False
    if prev_w_opt is not None and prev_lam_g is not None:
        warm_start = True
        coc.w_opt_prev = prev_w_opt
        coc.lam_g_prev = prev_lam_g
        coc.lam_x_prev = prev_lam_x

    # GLIDER_IPOPT_VERBOSE turns on IPOPT's own iteration log for diagnosis.
    res = coc.ocSolver(print_level=int(os.environ.get("GLIDER_IPOPT_VERBOSE", "0")),
                       horizon=task_horizon or config.ocp.horizon, 
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

        # Warm-start seed per initial condition, kept separate from last_trajs: a
        # "successful" IPOPT solve can still land in a much worse local optimum than
        # before, and warm-starting the next solve from it tends to lock the solver
        # into that same bad basin indefinitely. Only accept a new solution into the
        # warm-start reservoir if it isn't a large regression from the last accepted one.
        self.warm_start_trajs: Dict[Any, Dict[str, Any]] = {}
        self._warm_start_path = config.ocp.warm_start_path
        if self._warm_start_path and os.path.exists(self._warm_start_path):
            try:
                with open(self._warm_start_path, "rb") as fh:
                    self.warm_start_trajs = pickle.load(fh)
                self.logger.info(
                    f"Loaded {len(self.warm_start_trajs)} warm-start trajectories "
                    f"from {self._warm_start_path}")
            except Exception as exc:
                self.logger.warning(f"Could not load warm starts: {exc}")
        self._warm_start_regression_factor = 1.5

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        weights_CL = downstream_info["phi_CL"].view(-1, 1).detach().cpu().numpy()
        weights_CD = downstream_info["phi_CD"].view(-1, 1).detach().cpu().numpy()
        weights_CM = downstream_info["phi_CM"].view(-1, 1).detach().cpu().numpy()
        
        auxvar_vector = np.vstack([weights_CL, weights_CD, weights_CM])
        wing_geometry = downstream_info.get("wing_geometry")
        iteration = downstream_info["iteration"]

        tasks = self.config.task_list
        # (task, index-within-task) for every solve this iteration; all tasks share one
        # pool so the wall-clock cost of a second task is parallel, not serial.
        jobs = [(t, i, init_state) for t in tasks for i, init_state in enumerate(t.initial_states)]
        num_states = len(jobs)

        worker_args = []
        for task, i, init_state in jobs:
            prev = self.warm_start_trajs.get((task.name, i))
            worker_args.append((
                self.config, task.mode.value, task.horizon, task.stage_control_weight,
                init_state, auxvar_vector, wing_geometry,
                (prev or {}).get("w_opt"), (prev or {}).get("lam_g"), (prev or {}).get("lam_x"),
            ))

        num_workers = min(mp.cpu_count(), len(worker_args))
        if num_workers < 1: num_workers = 1
        
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.starmap(solve_worker, worker_args)

        # Energy audit: an unpowered glider cannot gain specific energy. A solve that
        # violates this is dynamically infeasible however IPOPT labelled it, so drop it
        # exactly as a failed solve is dropped rather than let it skew the objective.
        tol = getattr(self.config.ocp, "energy_gain_tol", 0.05)
        if tol and tol > 0:
            for (task, li, _), traj in zip(jobs, results):
                if not traj.get("success"):
                    continue
                st = traj["state_traj_opt"]
                e = st[:, 1] + (st[:, 4] ** 2 + st[:, 5] ** 2) / (2 * 9.81)
                gain = float(np.max(e) - e[0])
                if gain > tol:
                    traj["success"] = False
                    traj["return_status"] = f"EnergyGain({gain:.1f}m)"
                    self.logger.warning(
                        f"[{task.name}/{li}] rejected: specific energy rose {gain:.2f} m "
                        f"(tol {tol}) -- trajectory is not dynamically feasible")

        self.last_jobs = jobs
        self.last_trajs = results
        self.trajs_by_task = {t.name: [] for t in tasks}
        for (task, _, _), res in zip(jobs, results):
            self.trajs_by_task[task.name].append(res)

        def _fmt_cost(traj):
            try:
                return float(traj["cost"][0][0])
            except Exception:
                return float("nan")

        accepted = [False] * len(results)
        for i, traj in enumerate(results):
            if not traj["success"]:
                continue
            cost = _fmt_cost(traj)
            key = (jobs[i][0].name, jobs[i][1])
            prev = self.warm_start_trajs.get(key)
            # Tolerate a regression of (factor - 1) times the magnitude of the previous
            # cost. Scaling by abs() rather than multiplying the cost directly is what
            # keeps this a *loosened* bound for objectives that go negative -- a reward
            # like the glide distance -- where `factor * prev` would instead demand a
            # 50% improvement and freeze the reservoir at the first accepted solve.
            prev_cost = _fmt_cost(prev) if prev is not None else None
            # Always accept on the first iteration. The filter exists to reject a bad
            # solve of the *same* problem, but a reservoir loaded from disk may come
            # from a different design: a chain link seeded with a thick section
            # inherits a thin section's trajectories, whose cost (-40 m of glide) no
            # thick-wing solve (-5 m) can match, so every solve is rejected as a
            # regression and the stale guess is served forever. Bootstrapping once
            # re-anchors the reservoir on the design actually being solved.
            bootstrap = iteration == 0
            if prev is None or bootstrap or cost <= prev_cost + abs(prev_cost) * (self._warm_start_regression_factor - 1.0):
                self.warm_start_trajs[key] = traj
                accepted[i] = True

        if self._warm_start_path:
            # Only w_opt/lam_g/lam_x are read back as an initial guess; the rest of a
            # trajectory (sensitivity blocks and friends) is ~85 MB apiece, so keeping
            # it would mean rewriting ~700 MB every single iteration.
            try:
                slim = {k: {f: v[f] for f in ("w_opt", "lam_g", "lam_x") if f in v}
                        for k, v in self.warm_start_trajs.items()}
                tmp = self._warm_start_path + ".tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(slim, fh)
                os.replace(tmp, self._warm_start_path)
            except Exception as exc:
                self.logger.warning(f"Could not persist warm starts: {exc}")

        per_traj_summary = ", ".join(
            f"[{jobs[i][0].name}/{jobs[i][1]}] {'ok' if t['success'] else 'FAIL'} cost={_fmt_cost(t):.4f}"
            + ("" if t["success"] else f" [{t.get('return_status')}]")
            + (" (warm-start updated)" if accepted[i] else " (warm-start kept previous)" if t["success"] else "")
            for i, t in enumerate(self.last_trajs)
        )
        self.logger.info("per-trajectory: %s", per_traj_summary)
        if self.config.io.wandb.enabled:
            traj_metrics = {}
            for i, t in enumerate(self.last_trajs):
                tag = f"{jobs[i][0].name}_{jobs[i][1]}"
                traj_metrics[f"ocp/cost_traj_{tag}"] = _fmt_cost(t)
                traj_metrics[f"ocp/success_traj_{tag}"] = float(t["success"])
            wandb.log(traj_metrics, step=iteration)

        successful = [t for t in self.last_trajs if t["success"]]
        if successful:
            self.logger.info(
                "position error: %.6f",
                np.mean([np.linalg.norm(t["state_traj_opt"][-1][:2]) for t in successful])
            )
            self.logger.info(
                "velocity error: %.6f",
                np.mean([np.linalg.norm(t["state_traj_opt"][-1][4:6]) for t in successful])
            )
        else:
            self.logger.critical("All IPOPT solves failed; skipping position/velocity error logging")

        failures = num_states - len(successful)
        if failures > 0:
            self.logger.warning(f"{failures}/{num_states} IPOPT solves failed")
            
        num_iterations = self.config.run.max_outer_iters
        self._it = iteration
        log_every = self.config.io.log_every
        
        if iteration % log_every == 0 or iteration == (num_iterations - 1):
            self.plot_static(iteration)
            
        for i, traj in enumerate(self.last_trajs):
            np.save(f"traj_{jobs[i][0].name}_{jobs[i][1]}_{iteration}.npy", traj['state_traj_opt'])
            
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
            "trajectory_by_task": self.trajs_by_task,
            "iteration": downstream_info["iteration"],
            "augmented_lagrangian": downstream_info["augmented_lagrangian"] 
        }
    
    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        delta = 0.00001
        # {task_name: [dPhi_j/deps per trajectory]} together with the scalar weight the
        # augmented Lagrangian puts on that task (1 for the primary objective, and
        # d psi / d Phi_j = relu(lam_j + rho_j (Phi_j - eps_j)) for each requirement).
        dJ_deps_by_task = upstream_grads["dJ_deps_by_task"]
        task_weights = upstream_grads["task_weights"]

        # Each Phi_j is a mean over that task's own trajectories, so the per-trajectory
        # weight is the task weight divided by that task's successful count -- averaging
        # globally would let a task with more launch conditions dominate the gradient.
        successes = {}
        for (task, _, _), traj in zip(self.last_jobs, self.last_trajs):
            successes[task.name] = successes.get(task.name, 0) + int(bool(traj["success"]))

        dJ_deps_list = []
        for task, i, _ in self.last_jobs:
            n = successes.get(task.name, 0)
            scale = task_weights[task.name] / n if n else 0.0
            dJ_deps_list.append(scale * dJ_deps_by_task[task.name][i])

        total_dJ_dphi_np = None
        num_success = 0
        per_task_norm, per_task_vec = {}, {}

        for i, traj in enumerate(self.last_trajs):
            if not traj["success"]:
                continue

            dJ_deps = dJ_deps_list[i]

            auxsys_COC = traj['auxsys_COC']
            idoc_ctx = build_blocks_idoc(auxsys_COC, delta)
            traj_deriv_COC = idoc_full(idoc_ctx)

            deps_dphi = traj_deriv_COC['state_traj_opt']

            dJ_dphi_partial = np.einsum('ij,ijk->k', dJ_deps, deps_dphi).reshape(2028,1)

            if total_dJ_dphi_np is None:
                total_dJ_dphi_np = dJ_dphi_partial
            else:
                total_dJ_dphi_np += dJ_dphi_partial
            num_success += 1
            # per-task contribution to the design gradient, so the objective and the
            # requirement can be compared directly rather than inferred from weights
            tname = self.last_jobs[i][0].name
            per_task_norm[tname] = per_task_norm.get(tname, 0.0) + float(
                np.linalg.norm(dJ_dphi_partial))
            per_task_vec[tname] = (dJ_dphi_partial if tname not in per_task_vec
                                   else per_task_vec[tname] + dJ_dphi_partial)

        if num_success == 0:
            self.logger.critical("All IPOPT solves failed this iteration; skipping airfoil gradient update")
            total_dJ_dphi_np = np.zeros((2028, 1))

        if per_task_norm:
            parts = "  ".join(f"{k}:||g||={v:.3e}" for k, v in per_task_norm.items())
            cos = ""
            if len(per_task_vec) == 2:
                (na, va), (nb, vb) = list(per_task_vec.items())
                dot = float((va * vb).sum())
                den = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
                cos = f"  cos({na},{nb})={dot/den:+.3f}"
            self.logger.info(f"[GRAD] {parts}{cos}")

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
        run_name = getattr(self.config.io, "run_name", "run")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        task_names = list(dict.fromkeys(t.name for t, _, _ in getattr(self, "last_jobs", [])))
        task_colors = dict(zip(task_names, plt.cm.tab10(np.linspace(0, 1, max(len(task_names), 2)))))

        for (task, _, _), traj in zip(getattr(self, "last_jobs", []), self.last_trajs):
            states = traj['state_traj_opt']
            color = task_colors[task.name]
            ax.plot(states[:, 0], states[:, 1], color=color, alpha=0.7, linewidth=1.5, zorder=1)
            ax.scatter(states[0, 0], states[0, 1], color=color, marker='o', s=40, zorder=2)
            
        legend_elements = [
            Line2D([0], [0], color=task_colors[n], lw=2, label=n) for n in task_names
        ] + [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Start'),
        ]

        if any(t.mode in (EvaluationMode.Distance, EvaluationMode.ShortLanding)
               for t in self.config.task_list):
            # No target to reach here: the task is how far the glider gets before
            # touching down, so mark the ground instead.
            ax.axhline(0.0, color='saddlebrown', lw=2, zorder=0)
            legend_elements.append(Line2D([0], [0], color='saddlebrown', lw=2, label='Ground'))
        else:
            ax.scatter(0, 0, color='red', marker='x', s=100, linewidth=3, zorder=3)
            legend_elements.append(
                Line2D([0], [0], marker='x', color='red', markersize=10, markeredgewidth=2, label='Target')
            )
        ax.legend(handles=legend_elements, loc='upper right')
        
        title = f"All Trajectories - Iteration {iteration}"
        parts = []
        for name in task_names:
            d = [t['state_traj_opt'][-1, 0] - t['state_traj_opt'][0, 0]
                 for (tk, _, _), t in zip(self.last_jobs, self.last_trajs)
                 if tk.name == name and t["success"]]
            if d:
                parts.append(f"{name} {np.mean(d):.2f} m")
        if parts:
            title += " - " + ", ".join(parts)
        ax.set_title(title)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Z Position (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if self.config.io.wandb.enabled:
            wandb.log(
                {f"trajectory/all_trajectories_static": wandb.Image(fig)},
                step=iteration
            )
        else:
            out_dir = Path(self.config.io.checkpoint_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            static_plot_path = out_dir / f"{run_name}_static_iter{iteration}.png"
            fig.savefig(static_plot_path)
            
        plt.close(fig)

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