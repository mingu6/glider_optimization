from ..blockBase import Block
from typing import override
from ..config import Config, EvaluationMode
from ..utils.arm_jinenv import ArmThrowing
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import wandb
import logging

EVALUATION = "Trajectory"

class Evaluation(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.objective_evolution = []
        self.cost_evolution = []
        self.last_traj = None
        self._arm_env = None
        self.eval_map = {
            EvaluationMode.Perching: {
                "fwd": self.forward_ocp_cost,
                "bwd": self.backward_ocp_cost
            },
            EvaluationMode.SoftLanding: {
                "fwd": self.forward_ocp_cost,
                "bwd": self.backward_ocp_cost
            },
            EvaluationMode.Time: {
                "fwd": self.forward_time,
                "bwd": self.backward_time
            },
            EvaluationMode.RobotThrowing: {
                "fwd": self.forward_throw,
                "bwd": self.backward_throw
            }
        }
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self.last_traj = downstream_info["trajectory"]
        eval_mode = self.config.evaluation.mode
        eval_fn = self.eval_map[eval_mode]["fwd"]
        J = eval_fn()
                    
        aug = downstream_info["augmented_lagrangian"]
        total_obj = J + float(aug)
        
        iteration = downstream_info["iteration"]
        if iteration % self.config.io.log_every == 0:
            self.logger.info(f"Objective (total) = {total_obj}, Cost = {J}")
        
        if self.config.io.wandb.enabled:
            self._log_to_wandb(total_obj, J, aug, iteration)
        else:
            self.objective_evolution.append(total_obj)
            self.cost_evolution.append(J)

        return {
            "total_obj": total_obj
        }
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        eval_mode = self.config.evaluation.mode
        eval_fn = self.eval_map[eval_mode]["bwd"]
        dJ_deps_list = eval_fn()

        grads = {"dJ_deps": dJ_deps_list}
        if eval_mode == EvaluationMode.RobotThrowing:
            # The throw objective sees the design twice: through the trajectory
            # and, because the release point is at the end of the arm, directly.
            grads["dJ_dphi_direct"] = self.direct_throw_gradients()
        return grads

    def forward_ocp_cost(self):
        cost_vals = [float(t["cost"][0][0]) for t in self.last_traj if t["success"]]
        if not cost_vals:
            self.logger.critical("All trajectories failed; objective is undefined this iteration")
            return float("nan")
        return sum(cost_vals) / len(cost_vals)

    def forward_time(self):
        total_time = [t["state_traj_opt"][:,7].sum() for t in self.last_traj if t["success"]]
        if not total_time:
            self.logger.critical("All trajectories failed; objective is undefined this iteration")
            return float("nan")
        return sum(total_time) / len(total_time)
    
    def backward_ocp_cost(self):
        w = self.config.ocp.terminal_state_weight
        
        dJ_deps_list = []
        for traj in self.last_traj:
            dJ_deps_traj = np.zeros(traj['state_traj_opt'].shape)
            eps_terminal = traj['state_traj_opt'][-1]
            
            # TODO: this works only because the target is (0,0..)
            dJ_deps_traj[-1, :] = 2 * (w * eps_terminal)
            
            dJ_deps_list.append(dJ_deps_traj)
            
        return dJ_deps_list
        
    def backward_time(self):
        dJ_deps_list = []
        for traj in self.last_traj:
            dJ_deps_traj = np.zeros(traj['state_traj_opt'].shape)
            dJ_deps_traj[:,7] = 1.
            dJ_deps_list.append(dJ_deps_traj)

        return dJ_deps_list

    def _env(self):
        """Arm env, built once and only for its miss/landing CasADi functions."""
        if self._arm_env is None:
            self._arm_env = ArmThrowing(self.config)
            self._arm_env.initDyn()
        return self._arm_env

    def forward_throw(self):
        """Throw accuracy traded against swing duration.

            J = w_miss * |landing - target|^2 + w_time * sum_k h_k

        The duration is read straight off the trajectory because the timestep h
        is one of the states. The miss depends on the state at release AND on
        the design itself, which is why the backward pass has two pieces:
        backward_throw for dJ/dstate and direct_throw_gradients for dJ/ddesign.
        """
        arm = self.config.arm
        env = self._env()

        objectives = []
        for traj in self.last_traj:
            if not traj["success"]:
                continue
            states = traj['state_traj_opt']
            miss_sq = float(env.miss_of_fn(states[-1], traj["aux"], traj["target"]))
            objectives.append(arm.outer_w_miss * miss_sq
                              + arm.outer_w_time * float(states[:, -1].sum()))

        if not objectives:
            self.logger.critical("All trajectories failed; objective is undefined this iteration")
            return float("nan")
        return sum(objectives) / len(objectives)

    def backward_throw(self):
        arm = self.config.arm
        env = self._env()

        dJ_deps_list = []
        for traj in self.last_traj:
            states = traj['state_traj_opt']
            dJ_deps_traj = np.zeros(states.shape)

            # d/dh of the duration term, at every node (h is constant along the
            # trajectory, so all of them carry the derivative)
            dJ_deps_traj[:, -1] += arm.outer_w_time

            dmiss = np.asarray(
                env.dmiss_dx_fn(states[-1], traj["aux"], traj["target"])
            ).ravel()
            dJ_deps_traj[-1, :] += arm.outer_w_miss * dmiss

            dJ_deps_list.append(dJ_deps_traj)

        return dJ_deps_list

    def direct_throw_gradients(self):
        """dJ/d(auxvar) at fixed trajectory, in the OCP's auxvar layout.

        Only the miss contributes: the duration term is a pure function of the
        state, and the torque capacities (the tail of the auxvar) act on the
        inner problem's feasible set, never on J itself.
        """
        arm = self.config.arm
        env = self._env()

        direct_list = []
        for traj in self.last_traj:
            states = traj['state_traj_opt']
            dmiss_daux = np.asarray(
                env.dmiss_daux_fn(states[-1], traj["aux"], traj["target"])
            ).ravel()
            direct_list.append(np.concatenate([
                arm.outer_w_miss * dmiss_daux,
                np.zeros(env.N_DOF),
            ]))

        return direct_list

    def _log_to_wandb(self, total_obj, cost_val, aug, iteration):
        metrics = {
            "evaluation/objective_total": total_obj,
            "evaluation/ocp_cost": cost_val,
            "evaluation/augmented_lagrangian": aug
        }
        wandb.log(metrics, step=iteration)

    def plot_objective(self):
        out_dir = self._get_output_directory()
        run_name = self._get_run_name()
        
        self._save_plot(
            self.objective_evolution,
            "Total Objective",
            "Total Optimization Progress",
            out_dir / f"{run_name}_objective_total.png"
        )
        
        self._save_plot(
            self.cost_evolution,
            "Cost (OCP)",
            "OCP Cost Progress",
            out_dir / f"{run_name}_objective_cost.png"
        )
        
        return out_dir / f"{run_name}_objective_total.png", out_dir / f"{run_name}_objective_cost.png"

    def _get_output_directory(self):
        out_dir = Path(self.config.io.checkpoint_dir) if hasattr(self.config, "io") else Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _get_run_name(self):
        return getattr(self.config.io, "run_name", "run") if hasattr(self.config, "io") else "run"

    def _save_plot(self, data, ylabel, title, filepath):
        plt.figure()
        plt.plot(data)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        self.logger.info(f"Saved plot to {filepath}")