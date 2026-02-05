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
        self.objective_evolution = []
        self.cost_evolution = []
        self.logger = logging
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:        
        self.last_traj = downstream_info["trajectory"]
        cost_val = float(self.last_traj["cost"][0][0])
        aug = downstream_info.get("augmented_lagrangian", 0.0)
        total_obj = cost_val + float(aug)
        
        iteration = downstream_info["iteration"]        
        if iteration % self.config.io.log_every == 0:
            self.logger.info(f"Objective (total) = {total_obj}, Cost = {cost_val}")
        
        if self.config.io.wandb.enabled:
            wandb_metrics = {
                "evaluation/objective_total": total_obj,
                "evaluation/ocp_cost": cost_val,
                "evaluation/augmented_lagrangian": aug
            }
            wandb.log(wandb_metrics, step=iteration)
        
        else:
            self.objective_evolution.append(total_obj)
            self.cost_evolution.append(cost_val)

        return {}
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        w = self.config.ocp.terminal_state_weight
        epsT = self.last_traj['state_traj_opt'][-1]
        
        dJ_deps = w*epsT 
        dJ_deps = 2*dJ_deps
        
        return {
            "dJ_deps": dJ_deps,
        }
        
    def plot_objective(self):
        out_dir = Path(self.config.io.checkpoint_dir) if hasattr(self.config, "io") else Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        run_name = getattr(self.config.io, "run_name", "run") if hasattr(self.config, "io") else "run"
        out_path_total = out_dir / f"{run_name}_objective_total.png"
        plt.figure()
        plt.plot(self.objective_evolution)
        plt.xlabel("Iteration")
        plt.ylabel("Total Objective")
        plt.title("Total Optimization Progress")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path_total, dpi=150)
        plt.close()
        self.logger.info(f"Saved total objective plot to {out_path_total}")

        out_path_cost = out_dir / f"{run_name}_objective_cost.png"
        plt.figure()
        plt.plot(self.cost_evolution)
        plt.xlabel("Iteration")
        plt.ylabel("Cost (OCP)")
        plt.title("OCP Cost Progress")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path_cost, dpi=150)
        plt.close()
        self.logger.info(f"Saved cost-only plot to {out_path_cost}")

        return out_path_total, out_path_cost