from pathlib import Path
from ..blockBase import Block
from typing import Dict, Any, override, List, Optional
from ..utils.go_safe_pdp import COCsys
from ..utils.glider_jinenv import GliderPerching
from ..utils.arm_jinenv import ArmThrowing
from ..utils.idoc_ineq import build_blocks_idoc, idoc_full
from ..config import Config, EvaluationMode
from casadi import pi, vertcat, DM, Function
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

def _discretize(env, scheme: str = "rk4"):
    """Explicit one-step integrator over the free timestep carried in the state."""
    f_fun = Function(
        "f_fun",
        [env.X, env.U, env.dyn_auxvar],
        [env.f]
    )

    X = env.X
    U = env.U
    P = env.dyn_auxvar
    dt = X[-1]

    if scheme == "euler":
        return X + dt * f_fun(X, U, P)

    if scheme == "rk2":
        k1 = f_fun(X, U, P)
        k2 = f_fun(X + dt*k1, U, P)
        return X + dt/2*(k1 + k2)

    k1 = f_fun(X, U, P)
    k2 = f_fun(X + 0.5*dt*k1, U, P)
    k3 = f_fun(X + 0.5*dt*k2, U, P)
    k4 = f_fun(X + dt*k3, U, P)

    return X + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def _attach_sensitivity(coc: COCsys, res: Dict[str, Any]) -> Dict[str, Any]:
    # This shouldn't be part of the forward, however CasaDi functions cannot be pickled.
    try:
        res['auxsys_COC'] = coc.getAuxSys(opt_sol=res, threshold=1e-5)
    except Exception as e:
        res['auxsys_COC'] = None
        print(f"Warning: Failed to extract sensitivity data: {e}")
    return res


def arm_initial_guess(config: Config, target=None) -> List[float]:
    """A swing-shaped guess for the NLP variables [X_0, U_0, ..., X_N].

    IPOPT's default guess here is the arm falling under gravity with zero
    torque, from which the throw is a long way off: cold solves wander for
    thousands of iterations and end up releasing the ball downwards. Sweeping
    the joints from the wind-up pose to a plausible release pose, with a
    sinusoidal rate profile (zero at both ends, so it is consistent with the
    trapezoid of the sweep), puts the solver in the right basin from the start.
    The shoulder azimuth is aimed straight at the target, which is the one part
    of the release pose that genuinely depends on where the throw is going.
    """
    arm = config.arm
    nd = len(arm.q_start)
    q_start = np.asarray(arm.q_start, dtype=float)
    q_end = np.asarray(arm.q_end_guess, dtype=float)
    if target is not None:
        q_end = q_end.copy()
        q_end[0] = float(np.arctan2(target[1], target[0]))
    h = arm.h_init
    N = arm.horizon

    guess: List[float] = []
    for k in range(N + 1):
        s = k / N
        q = (1.0 - s) * q_start + s * q_end
        qd = np.sin(np.pi * s) * (q_end - q_start) / (N * h)
        guess += list(q) + list(qd) + [0.0]*nd + [h]   # torque guess: unloaded
        if k < N:
            guess += [0.0] * nd                        # torque rates
    return guess


def _arm_target_stages(config: Config, target) -> List[List[float]]:
    """Targets to walk out to `target` from a throw that solves cold.

    The stages step in absolute distance along the direction of the target,
    starting at cold_start_first_distance. A fraction of the target would be
    the obvious parameterisation and is the wrong one: for a far target it
    starts with a very NEAR throw, which is a hard problem in its own right
    (the ball has to be lofted gently while the minimum-time term pushes the
    swing to go as fast as it can), and the whole chain then starts from a
    failed solve.
    """
    arm = config.arm
    tgt = np.asarray(target, dtype=float)
    distance = float(np.linalg.norm(tgt))
    stages = max(int(arm.cold_start_stages), 1)

    if stages == 1 or distance <= arm.cold_start_first_distance:
        return [list(target)]

    direction = tgt / distance
    return [list(direction * d)
            for d in np.linspace(arm.cold_start_first_distance, distance, stages)]


def _build_arm_coc(env, config: Config, target, X_next) -> COCsys:
    """The throw at one target, as a COCsys problem."""
    arm = config.arm

    env.initCost(target, w_miss=arm.w_miss, wu=arm.wu, w_time=arm.w_time,
                 stage_scale=arm.stage_scale, w_taudot=arm.w_taudot)
    env.initConstraints(arm.q_min, arm.q_max, caps=None, p_joint=arm.p_joint,
                        du_max=arm.du_max)

    state_lb, state_ub = env.state_bounds(arm.h_min, arm.h_max)

    coc = COCsys()
    coc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.constraint_auxvar))
    coc.setStateVariable(env.X, state_lb=state_lb, state_ub=state_ub)
    coc.setControlVariable(env.U)
    coc.setDyn(X_next)

    coc.setPathCost(env.path_cost)
    coc.setFinalCost(env.final_cost)

    coc.setPathInequCstr(env.path_inequ)
    coc.setFinalInequCstr(env.final_inequ)

    coc.diffCPMP()
    return coc


def solve_arm_worker(config: Config,
                     target: List[float],
                     psi: np.ndarray,
                     prev_w_opt: Optional[List[float]] = None,
                     prev_lam_g: Optional[List[float]] = None,
                     prev_lam_x: Optional[List[float]] = None) -> Dict[str, Any]:
    """Minimum-time throw at `target` for the fixed design `psi`.

    A cold solve walks the target out from a near one (see _arm_target_stages);
    a warm one goes straight at the real target from the previous outer
    iteration's solution.
    """
    arm = config.arm
    env = ArmThrowing(config)
    env.initDyn()

    # psi -> auxvar is the design map: the first block feeds the DYNAMICS, the
    # torque capacities feed the OCP's own CONSTRAINTS. Both halves are handed
    # to COCsys as one auxvar so the sensitivity covers both channels.
    aux = np.asarray(env.psi_to_auxvar(psi)).ravel()
    caps = np.asarray(env.psi_to_caps(psi)).ravel()
    auxvar_vector = np.concatenate([aux, caps]).reshape(-1, 1)

    X_next = _discretize(env, arm.integrator)
    # rates and torques both start at zero: the arm is at rest and unloaded
    init_state = list(arm.q_start) + [0.0]*(2*env.N_DOF) + [arm.h_init]

    warm = None
    if prev_w_opt is not None and prev_lam_g is not None:
        warm = (prev_w_opt, prev_lam_g, prev_lam_x)
        stage_targets = [list(target)]
    else:
        stage_targets = _arm_target_stages(config, target)

    for stage_idx, stage_target in enumerate(stage_targets):
        coc = _build_arm_coc(env, config, stage_target, X_next)

        if warm is not None:
            coc.w_opt_prev, coc.lam_g_prev, coc.lam_x_prev = warm

        solver_opts = dict(arm.ipopt_options)
        if stage_idx < len(stage_targets) - 1:
            solver_opts["ipopt.max_iter"] = arm.cold_start_stage_max_iter

        res = coc.ocSolver(horizon=arm.horizon,
                           init_state=init_state,
                           auxvar_value=auxvar_vector,
                           timeVarying=True,
                           warm_start=warm is not None,
                           init_guess=None if warm is not None
                           else arm_initial_guess(config, stage_target),
                           solver_opts=solver_opts)

        # A stage that failed is a poor seed for the next one; keep the last
        # good solution as the warm start instead.
        if res["success"]:
            warm = (res["w_opt"], res["lam_g"], res["lam_x"])

    res['target'] = list(target)
    res['aux'] = aux
    res['caps'] = caps

    return _attach_sensitivity(coc, res)


def solve_worker(config: Config,
                 init_state: List[float],
                 auxvar_vector: np.ndarray,
                 wing_geometry: Optional[Dict[str, Any]] = None,
                 prev_w_opt: Optional[List[float]] = None,
                 prev_lam_g: Optional[List[float]] = None,
                 prev_lam_x: Optional[List[float]] = None) -> Dict[str, Any]:

    env = GliderPerching(config, wing_geometry=wing_geometry)
    coc = COCsys()

    env.initDyn()
    X_next = _discretize(env, "rk4")

    env.initCost(state_weights=config.ocp.terminal_state_weight, wu=config.ocp.stage_control_weight, init_state=init_state)
    env.initConstraints(-pi/3, pi/8, 2.2)
    
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

    return _attach_sensitivity(coc, res)

class OCP(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.run.device)
        self.logger = logging
        self.is_arm = config.evaluation.mode == EvaluationMode.RobotThrowing

        # Not used to solve, kept only for plotting/animation purposes
        if self.is_arm:
            self.env = ArmThrowing(self.config)
            self.env.initDyn()
            # [L, m, com, Iax, Itr] for 3 segments, then one torque capacity per joint
            self.n_auxvar = 5 * self.env.N_SEG + self.env.N_DOF
        else:
            self.env = GliderPerching(self.config)
            self.env.state_weights = config.ocp.terminal_state_weight
            self.n_auxvar = 3 * (config.reducedModel.chebyshev_degree + 1) ** 2

        self.last_trajs: List[Dict[str, Any]] = []

        self.warm_start_trajs: List[Optional[Dict[str, Any]]] = []
        self._warm_start_regression_factor = 1.5

    def _warm_start_for(self, i: int):
        if len(self.warm_start_trajs) > i and self.warm_start_trajs[i] is not None:
            prev_sol = self.warm_start_trajs[i]
            return prev_sol.get("w_opt"), prev_sol.get("lam_g"), prev_sol.get("lam_x")
        return None, None, None

    def _build_worker_args(self, downstream_info: Dict[str, Any]):
        """(worker function, per-scenario argument tuples) for this problem."""
        if self.is_arm:
            psi = downstream_info["params"].detach().cpu().numpy().ravel()
            self._last_psi = psi
            args = [
                (self.config, target, psi, *self._warm_start_for(i))
                for i, target in enumerate(self.config.arm.targets)
            ]
            return solve_arm_worker, args

        weights_CL = downstream_info["phi_CL"].view(-1, 1).detach().cpu().numpy()
        weights_CD = downstream_info["phi_CD"].view(-1, 1).detach().cpu().numpy()
        weights_CM = downstream_info["phi_CM"].view(-1, 1).detach().cpu().numpy()

        auxvar_vector = np.vstack([weights_CL, weights_CD, weights_CM])
        wing_geometry = downstream_info.get("wing_geometry")

        args = [
            (self.config, init_state, auxvar_vector, wing_geometry, *self._warm_start_for(i))
            for i, init_state in enumerate(self.config.ocp.initial_states)
        ]
        return solve_worker, args

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()

        iteration = downstream_info["iteration"]

        worker_fn, worker_args = self._build_worker_args(downstream_info)
        num_states = len(worker_args)

        num_workers = min(mp.cpu_count(), len(worker_args))
        if num_workers < 1: num_workers = 1

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.starmap(worker_fn, worker_args)

        self.last_trajs = results

        def _fmt_cost(traj):
            try:
                return float(traj["cost"][0][0])
            except Exception:
                return float("nan")

        if len(self.warm_start_trajs) < len(results):
            self.warm_start_trajs.extend([None] * (len(results) - len(self.warm_start_trajs)))
        accepted = [False] * len(results)
        for i, traj in enumerate(results):
            if not traj["success"]:
                continue
            cost = _fmt_cost(traj)
            prev = self.warm_start_trajs[i]
            if prev is None or cost <= self._warm_start_regression_factor * _fmt_cost(prev):
                self.warm_start_trajs[i] = traj
                accepted[i] = True

        per_traj_summary = ", ".join(
            f"[{i}] {'ok' if t['success'] else 'FAIL'} cost={_fmt_cost(t):.4f}"
            + (" (warm-start updated)" if accepted[i] else " (warm-start kept previous)" if t["success"] else "")
            for i, t in enumerate(self.last_trajs)
        )
        self.logger.info("per-trajectory: %s", per_traj_summary)
        if self.config.io.wandb.enabled:
            traj_metrics = {}
            for i, t in enumerate(self.last_trajs):
                traj_metrics[f"ocp/cost_traj_{i}"] = _fmt_cost(t)
                traj_metrics[f"ocp/success_traj_{i}"] = float(t["success"])
            wandb.log(traj_metrics, step=iteration)

        successful = [t for t in self.last_trajs if t["success"]]
        if successful:
            self._log_solution_metrics(successful, iteration)
        else:
            self.logger.critical("All IPOPT solves failed; skipping solution metrics")

        failures = num_states - len(successful)
        if failures > 0:
            self.logger.warning(f"{failures}/{num_states} IPOPT solves failed")

        num_iterations = self.config.run.max_outer_iters
        self._it = iteration
        log_every = self.config.io.log_every

        if iteration % log_every == 0 or iteration == (num_iterations - 1):
            self.plot_static(iteration)

        if self.is_arm:
            if iteration % log_every == 0 or iteration == (num_iterations - 1):
                self._save_arm_solutions(iteration)
            if self.config.arm.animate and iteration in (0, num_iterations - 1):
                self.plot_animations(iteration)
        else:
            for i, traj in enumerate(self.last_trajs):
                np.save(f"traj_{i}_{iteration}.npy", traj['state_traj_opt'])

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
    
    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        delta = 0.00001
        dJ_deps_list = upstream_grads["dJ_deps"]
        # Present when the objective depends on the auxvar other than through the
        # trajectory (the arm's release point moves with the segment lengths).
        direct_list = upstream_grads.get("dJ_dphi_direct")

        total_dJ_dphi_np = None
        num_success = 0

        for i, traj in enumerate(self.last_trajs):
            if not traj["success"]:
                continue

            dJ_deps = dJ_deps_list[i]

            auxsys_COC = traj['auxsys_COC']
            idoc_ctx = build_blocks_idoc(auxsys_COC, delta)
            try:
                traj_deriv_COC = idoc_full(idoc_ctx)
            except np.linalg.LinAlgError as e:
                # A degenerate active set (linearly dependent active constraints)
                # makes A H^-1 A^T singular; delta cannot regularize that away,
                # since it only conditions H. Drop this trajectory's gradient
                # rather than lose the whole run.
                self.logger.error(
                    "trajectory %d: IDOC failed (%s); skipping its gradient", i, e
                )
                continue

            deps_dphi = traj_deriv_COC['state_traj_opt']

            dJ_dphi_partial = np.einsum('ij,ijk->k', dJ_deps, deps_dphi).reshape(self.n_auxvar, 1)

            if direct_list is not None:
                dJ_dphi_partial = dJ_dphi_partial + np.asarray(direct_list[i]).reshape(self.n_auxvar, 1)

            if total_dJ_dphi_np is None:
                total_dJ_dphi_np = dJ_dphi_partial
            else:
                total_dJ_dphi_np += dJ_dphi_partial
            num_success += 1

        if num_success == 0:
            self.logger.critical("All IPOPT solves failed this iteration; skipping design gradient update")
            total_dJ_dphi_np = np.zeros((self.n_auxvar, 1))
        else:
            total_dJ_dphi_np /= num_success

        dJ_dphi = torch.from_numpy(total_dJ_dphi_np).float().to(self.device)
        # glider: (CL, CD, CM) blocks of Chebyshev coefficients. arm: one flat auxvar.
        dJ_dphi = dJ_dphi.view(-1) if self.is_arm else dJ_dphi.view(3, -1)

        end_time = time.perf_counter()
        backward_time = end_time - start_time
                
        if self.config.io.wandb.enabled:
            wandb.log({
                "profiler/ocp_backward_time": backward_time,
            }, step=self._it)

        return {"dJ_dphi": dJ_dphi}
                
    def _log_solution_metrics(self, successful, iteration):
        """Problem-specific readout of what the inner solves achieved."""
        if not self.is_arm:
            self.logger.info(
                "position error: %.6f",
                np.mean([np.linalg.norm(t["state_traj_opt"][-1][:2]) for t in successful])
            )
            self.logger.info(
                "velocity error: %.6f",
                np.mean([np.linalg.norm(t["state_traj_opt"][-1][4:6]) for t in successful])
            )
            return

        horizon = self.config.arm.horizon
        misses, durations, speeds = [], [], []
        for t in successful:
            states = t["state_traj_opt"]
            misses.append(self._arm_miss(t))
            durations.append(float(states[0, -1]) * horizon)
            _, v = self.env.tip_fn(states[-1], t["aux"])
            speeds.append(float(np.linalg.norm(np.asarray(v).ravel())))

        self.logger.info("miss: %.6f m", float(np.mean(misses)))
        self.logger.info("swing duration: %.4f s", float(np.mean(durations)))
        self.logger.info("release speed: %.3f m/s", float(np.mean(speeds)))

        if self.config.io.wandb.enabled:
            wandb.log({
                "ocp/miss_mean": float(np.mean(misses)),
                "ocp/miss_max": float(np.max(misses)),
                "ocp/swing_duration_mean": float(np.mean(durations)),
                "ocp/release_speed_mean": float(np.mean(speeds)),
            }, step=iteration)

    def _arm_miss(self, traj) -> float:
        """Distance between where the ball lands and the target, in metres."""
        land = np.asarray(self.env.landing_fn(traj["state_traj_opt"][-1], traj["aux"])[0]).ravel()
        return float(np.linalg.norm(land - np.asarray(traj["target"], dtype=float)))

    def _arm_ball_flight(self, traj, n=60):
        """Ballistic path of the ball released at the final node."""
        states = traj["state_traj_opt"]
        p, v = self.env.tip_fn(states[-1], traj["aux"])
        p = np.asarray(p).ravel()
        v = np.asarray(v).ravel()
        t_fly = float(np.asarray(self.env.landing_fn(states[-1], traj["aux"])[1]).ravel()[0])
        tb = np.linspace(0.0, max(t_fly, 1e-3), n)
        return np.stack([p[0] + v[0]*tb,
                         p[1] + v[1]*tb,
                         p[2] + v[2]*tb - 0.5*9.81*tb**2], axis=1)

    def _save_arm_solutions(self, iteration):
        """Dump each solve in the format playground/mj_render.py consumes.

        Everything the renderer needs is written out, including the release
        state and the arm's geometry, so it never has to rebuild an env and
        cannot go stale when the state layout changes.
        """
        out_dir = Path(self.config.io.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, traj in enumerate(self.last_trajs):
            if not traj["success"]:
                continue

            states = traj["state_traj_opt"]
            p, v = self.env.tip_fn(states[-1], traj["aux"])
            land, t_fly = self.env.landing_fn(states[-1], traj["aux"])

            np.savez(out_dir / f"throw_iter{iteration}_target{i}.npz",
                     state=states,
                     control=traj["control_traj_opt"],
                     aux=traj["aux"],
                     caps=traj["caps"],
                     psi=self._last_psi,
                     target=np.asarray(traj["target"], dtype=float),
                     release_p=np.asarray(p).ravel(),
                     release_v=np.asarray(v).ravel(),
                     t_fly=float(np.asarray(t_fly).ravel()[0]),
                     land=np.asarray(land).ravel(),
                     lengths=np.asarray(traj["aux"]).ravel()[:self.env.N_SEG],
                     shoulder_h=float(self.env.shoulder_h),
                     z_target=float(self.env.z_target),
                     r0=np.asarray(self.env.R0, dtype=float))

    def plot_static(self, iteration):
        if self.is_arm:
            return self.plot_static_arm(iteration)

        run_name = getattr(self.config.io, "run_name", "run")

        fig, ax = plt.subplots(figsize=(10, 8))
        
        num_trajs = len(self.last_trajs)
        colors = plt.cm.viridis(np.linspace(0, 1, num_trajs))
        
        for i, traj in enumerate(self.last_trajs):                
            states = traj['state_traj_opt']
            
            ax.plot(states[:, 0], states[:, 1], color=colors[i], alpha=0.7, linewidth=1.5, zorder=1)
            
            ax.scatter(states[0, 0], states[0, 1], color=colors[i], marker='o', s=40, zorder=2)
            
        ax.scatter(0, 0, color='red', marker='x', s=100, linewidth=3, zorder=3)
        
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, label='Trajectories'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Start'),
            Line2D([0], [0], marker='x', color='red', markersize=10, markeredgewidth=2, label='Target')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f"All Trajectories - Iteration {iteration}")
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

    def plot_static_arm(self, iteration):
        """One figure per outer iteration: every throw in 3D, plus the joint
        traces of the first successful one against the limits the design sets."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        run_name = getattr(self.config.io, "run_name", "run")
        arm = self.config.arm
        nd = self.env.N_DOF

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1.0], hspace=0.5,
                              wspace=0.32, left=0.02, right=0.96,
                              top=0.92, bottom=0.08)
        ax = fig.add_subplot(gs[0:2, 0], projection='3d')
        ax_side = fig.add_subplot(gs[2, 0])

        colors = plt.cm.viridis(np.linspace(0, 0.85, max(len(self.last_trajs), 1)))
        pts = [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, self.env.shoulder_h]])]
        side_pts = []

        for i, traj in enumerate(self.last_trajs):
            target = np.asarray(traj["target"], dtype=float)
            ax.scatter([target[0]], [target[1]], [self.env.z_target], s=120,
                       c=[colors[i]], marker='v', depthshade=False, zorder=10)
            if not traj["success"]:
                continue

            states = traj['state_traj_opt']
            joints = np.array([self.env.joint_positions(s[:nd], traj["aux"])
                               for s in states])
            ball = self._arm_ball_flight(traj)
            pts += [joints.reshape(-1, 3), ball]
            side_pts.append(joints.reshape(-1, 3)[:, [0, 2]])

            # stroboscopic arm: a handful of poses through the swing, opaque at release
            strobe = np.linspace(0, len(states) - 1, 6).astype(int)
            for k in strobe:
                P = joints[k]
                ax.plot(P[:, 0], P[:, 1], P[:, 2], '-o', lw=3 if k == len(states)-1 else 1.6,
                        markersize=3, color=colors[i],
                        alpha=1.0 if k == len(states)-1 else 0.25, zorder=6)
                ax_side.plot(P[:, 0], P[:, 2], '-o', lw=3 if k == len(states)-1 else 1.4,
                             markersize=3, color=colors[i],
                             alpha=1.0 if k == len(states)-1 else 0.25, zorder=3)

            ax.plot(joints[:, -1, 0], joints[:, -1, 1], joints[:, -1, 2],
                    '-', lw=2, color='#F77F00', alpha=0.9, zorder=5)
            ax.plot(ball[:, 0], ball[:, 1], ball[:, 2], '--', lw=2,
                    color=colors[i], alpha=0.9, zorder=5)
            ax_side.plot(joints[:, -1, 0], joints[:, -1, 2], '-', lw=2,
                         color='#F77F00', alpha=0.9, zorder=4)

        ax.plot([0, 0], [0, 0], [0, self.env.shoulder_h], color='#555', lw=3, alpha=0.7)

        allp = np.vstack(pts)
        ax.set_xlim(min(allp[:, 0].min() - 0.3, -0.5), allp[:, 0].max() + 0.3)
        ax.set_ylim(min(allp[:, 1].min() - 0.5, -1.0), max(allp[:, 1].max() + 0.5, 1.0))
        ax.set_zlim(0, max(allp[:, 2].max() + 0.3, 2.0))
        ax.set_box_aspect((2.6, 1.3, 1.0))
        ax.view_init(elev=16, azim=-62)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)", labelpad=-2)
        ax.tick_params(pad=-1)
        ax.set_title(f"Throws - Iteration {iteration}", fontweight='bold')

        legend_elements = [
            Line2D([0], [0], color='#F77F00', lw=2, label='Hand path'),
            Line2D([0], [0], color='gray', lw=2, ls='--', label='Ball flight'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='gray',
                   markersize=10, label='Target'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        # the throw is metres long and the arm is centimetres, so the 3D view
        # alone makes the actual motion unreadable: this is the same swing at
        # arm scale
        ax_side.plot([0, 0], [0, self.env.shoulder_h], color='#555', lw=3, alpha=0.7, zorder=1)
        if side_pts:
            sp = np.vstack(side_pts)
            ax_side.set_xlim(sp[:, 0].min() - 0.15, sp[:, 0].max() + 0.15)
            ax_side.set_ylim(sp[:, 1].min() - 0.15, sp[:, 1].max() + 0.15)
        # datalim, not box: the panel keeps its width and the view widens instead
        # of collapsing to a sliver around a 70 cm arm
        ax_side.set_aspect('equal', adjustable='datalim')
        ax_side.set_xlabel("X (m)", fontsize=9)
        ax_side.set_ylabel("Z (m)", fontsize=9)
        ax_side.set_title("swing, at arm scale", fontsize=10, fontweight='bold')
        ax_side.grid(True, alpha=0.3)

        shown = next((t for t in self.last_trajs if t["success"]), None)
        names = ['sh-az', 'sh-el', 'elbow', 'wrist']
        joint_colors = ['#E63946', '#06A77D', '#118AB2', '#9D4EDD']

        ax_q = fig.add_subplot(gs[0, 1])
        ax_qd = fig.add_subplot(gs[1, 1])
        ax_u = fig.add_subplot(gs[2, 1])

        if shown is not None:
            states = shown['state_traj_opt']
            caps = np.asarray(shown["caps"]).ravel()
            w_max = arm.p_joint / caps

            for j in range(nd):
                ax_q.plot(states[:, j], lw=2, color=joint_colors[j], label=names[j])
                ax_qd.plot(states[:, nd + j], lw=2, color=joint_colors[j])
                ax_qd.axhline(w_max[j], color=joint_colors[j], ls=':', lw=1, alpha=0.7)
                ax_qd.axhline(-w_max[j], color=joint_colors[j], ls=':', lw=1, alpha=0.7)
                # torque is a state; the control is its rate
                ax_u.plot(states[:, 2*nd + j], lw=2, color=joint_colors[j])
                ax_u.axhline(caps[j], color=joint_colors[j], ls=':', lw=1, alpha=0.7)
                ax_u.axhline(-caps[j], color=joint_colors[j], ls=':', lw=1, alpha=0.7)

            miss = self._arm_miss(shown)
            duration = float(states[0, -1]) * arm.horizon
            ax_q.set_title(f"first throw: {duration*1000:.0f} ms swing, miss {miss:.3f} m",
                           fontsize=10, fontweight='bold')
            ax_q.legend(fontsize=7, ncol=4, loc='upper left')
        else:
            ax_q.set_title("all solves failed", fontsize=10, fontweight='bold')

        ax_q.set_ylabel("q (rad)")
        ax_qd.set_ylabel("qd (rad/s)\ndotted: no-load")
        ax_u.set_ylabel("tau (N m)\ndotted: capacity")
        ax_u.set_xlabel("node")
        for a in (ax_q, ax_qd, ax_u):
            a.grid(True, alpha=0.3)

        if self.config.io.wandb.enabled:
            wandb.log({"trajectory/all_trajectories_static": wandb.Image(fig)},
                      step=iteration)
        else:
            out_dir = Path(self.config.io.checkpoint_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{run_name}_static_iter{iteration}.png", dpi=120)

        plt.close(fig)

    def plot_animations(self, iteration):
        if self.is_arm:
            return self.plot_animations_arm(iteration)

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

    def plot_animations_arm(self, iteration):
        run_name = getattr(self.config.io, "run_name", "run")

        for i, traj in enumerate(self.last_trajs):
            if not traj["success"]:
                continue

            # play_animation reads self.target / self.land_fn, which initCost binds
            self.env.initCost(traj["target"], w_miss=self.config.arm.w_miss,
                              wu=self.config.arm.wu, w_time=self.config.arm.w_time,
                              stage_scale=self.config.arm.stage_scale)

            suffix = f"_target{i}"
            if self.config.io.wandb.enabled:
                with tempfile.TemporaryDirectory() as tmpdir:
                    title = Path(tmpdir) / f"{run_name}_traj_iter{iteration}{suffix}"
                    self.env.play_animation(
                        traj['state_traj_opt'],
                        traj['control_traj_opt'],
                        traj['aux'],
                        save_option=True,
                        title=str(title),
                        fps=self.config.io.gif_fps,
                    )
                    wandb.log(
                        {f"trajectory/traj_iter_{iteration}_target{i}":
                            wandb.Video(f"{title}.gif", format="gif")},
                        step=iteration
                    )
            else:
                out_dir = Path(self.config.io.checkpoint_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                title = out_dir / f"{run_name}_traj_iter{iteration}{suffix}"
                self.env.play_animation(
                    traj['state_traj_opt'],
                    traj['control_traj_opt'],
                    traj['aux'],
                    save_option=True,
                    title=str(title),
                    fps=self.config.io.gif_fps
                )