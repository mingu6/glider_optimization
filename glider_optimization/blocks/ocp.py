from pathlib import Path
from ..blockBase import Block
from typing import Dict, Any, override, List, Optional
from ..utils.go_safe_pdp import COCsys
from ..utils.glider_jinenv import GliderPerching
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

def solve_worker(config: Config, 
                 init_state: List[float], 
                 auxvar_vector: np.ndarray, 
                 prev_w_opt: Optional[List[float]] = None, 
                 prev_lam_g: Optional[List[float]] = None, 
                 prev_lam_x: Optional[List[float]] = None) -> Dict[str, Any]:

    env = GliderPerching(config)
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
        
        weights_CL = downstream_info["phi_CL"].view(-1, 1).detach().cpu().numpy()
        weights_CD = downstream_info["phi_CD"].view(-1, 1).detach().cpu().numpy()
        weights_CM = downstream_info["phi_CM"].view(-1, 1).detach().cpu().numpy()
        
        auxvar_vector = np.vstack([weights_CL, weights_CD, weights_CM])
        
        initial_states = self.config.ocp.initial_states
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
            
            worker_args.append((self.config, init_state, auxvar_vector, prev_w, prev_lg, prev_lx))

        num_workers = min(mp.cpu_count(), len(worker_args))
        if num_workers < 1: num_workers = 1
        
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(solve_worker, worker_args)
            
        self.last_trajs = results
        
        failures = sum(1 for r in results if not r["success"])
        if failures > 0:
            self.logger.warning(f"⚠️ {failures}/{num_states} IPOPT solves failed")
            
        num_iterations = self.config.run.max_outer_iters
        iteration = downstream_info["iteration"]
        self._it = iteration
        log_every = self.config.io.log_every
        
        if iteration % log_every == 0 or iteration == (num_iterations - 1):
            self.plot_static(iteration)
            
        if iteration == 0 or iteration == (num_iterations - 1):
            self.plot_animations(iteration)

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
        
        total_dJ_dphi_np = None

        for i, traj in enumerate(self.last_trajs):
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
             
        total_dJ_dphi_np /= len(self.last_trajs)

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