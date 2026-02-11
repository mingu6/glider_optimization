from ..blockBase import Block
from typing import override
from ..config import Config
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
import logging


class Evaluation(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.objective_evolution = []
        self.cost_evolution = []
        self.last_traj = None
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self.last_traj = downstream_info["trajectory"]
        
        cost_vals = [float(t["cost"][0][0]) for t in self.last_traj]
        cost_val = sum(cost_vals) / len(cost_vals)
            
        aug = downstream_info["augmented_lagrangian"]
        total_obj = cost_val + float(aug)
        
        iteration = downstream_info["iteration"]
        if iteration % self.config.io.log_every == 0:
            self.logger.info(f"Objective (total) = {total_obj}, Cost = {cost_val}")
        
        if self.config.io.wandb.enabled:
            self._log_to_wandb(total_obj, cost_val, aug, iteration)
        else:
            self.objective_evolution.append(total_obj)
            self.cost_evolution.append(cost_val)

        return {}
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        w = self.config.ocp.terminal_state_weight
        
        dJ_deps_list = []
        for traj in self.last_traj:
            eps_terminal = traj['state_traj_opt'][-1]
            dJ_deps_list.append(2 * (w * eps_terminal))
        return {"dJ_deps": dJ_deps_list}

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