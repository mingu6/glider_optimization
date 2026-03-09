from ..blockBase import Block
# from typing import override
from typing_extensions import override
from ..config import Config, EvaluationMode
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
            "total_obj": total_obj,
            "cost": J,
            "augmented_lagrangian": float(aug)
        }
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:        
        eval_mode = self.config.evaluation.mode
        eval_fn = self.eval_map[eval_mode]["bwd"]
        dJ_deps_list = eval_fn()
        
        return {"dJ_deps": dJ_deps_list}

    def forward_ocp_cost(self):
        
        cost_vals = [float(t["cost"][0][0]) for t in self.last_traj]
        return sum(cost_vals) / len(cost_vals)
        
    def forward_time(self):
        total_time = [t["state_traj_opt"][:,7].sum() for t in self.last_traj]
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