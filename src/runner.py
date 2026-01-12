from typing import Any, Dict
from pathlib import Path
from .config import Config
from .blockBase import Block
from .blocks import Airfoil, NeuralFoilSampling, ReducedModel, Evaluation, OCP
import logging
import matplotlib
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        # The order matters! 
        # The output of block i-1 is the input of block i in the forward phase
        # The output of block i is the input of block i-1 in the backward phase
        self.blocks : Dict[str, Block] = {
            "Airfoil": Airfoil(config),
            "NeuralFoilSampling": NeuralFoilSampling(config),
            "ReducedModel": ReducedModel(config),
            "OCP": OCP(config)
        }
        self.objective_evolution = []
        self.cost_evolution = []
        self.setup_environment()

    def setup_environment(self):
        seed = self.config.run.seed
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        self.logger.info(f"Environment initialized with seed {seed}")

    def run(self):
        self.logger.info("Runner started")
        num_iterations = self.config.run.max_outer_iters

        for iteration in range(num_iterations):
            if iteration % self.config.io.log_every == 0:
                self.logger.info("="*100)
                self.logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            self.forward_pass(iteration)
            self.backward_pass(iteration)

            if iteration == 0 or iteration == (num_iterations - 1):
                ocp_block = self.blocks.get("OCP")
                if ocp_block is not None and hasattr(ocp_block, 'last_traj_COC') and ocp_block.last_traj_COC is not None:
                    out_dir = Path(self.config.io.checkpoint_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    run_name = getattr(self.config.io, "run_name", "run")
                    title = out_dir / f"{run_name}_traj_iter{iteration}"
                    try:
                        ocp_block.env.play_animation(ocp_block.last_traj_COC['state_traj_opt'], ocp_block.last_traj_COC['control_traj_opt'], save_option=True, title=str(title), fps=self.config.io.gif_fps)
                    except Exception as e:
                        self.logger.error(f"Failed to save trajectory animation for iter {iteration}: {e}")
        
        self.blocks["Airfoil"].save_gif(fps=self.config.io.gif_fps)
        self.plot_objective()
        self.logger.info("Runner finished")
        
    def checkpoint_on_interrupt(): ... # TODO

    def forward_pass(self, it):
        self.logger.debug("Forward pass started")
        
        propagationDict = {"iteration": it}
        
        for block_name, block in self.blocks.items():
            self.logger.debug("Forward block "+block_name)
            propagationDict = block.forward(propagationDict)
        
        obj = propagationDict.get("objective")
        cost_only = propagationDict.get("cost", None)

        if it % self.config.io.log_every == 0:
            if cost_only is not None:
                self.logger.info(f"Objective (total) = {obj}, Cost = {cost_only}")
            else:
                self.logger.info(f"Objective = {obj}")

        self.objective_evolution.append(obj.item() if hasattr(obj, "item") else obj)
        if cost_only is not None:
            self.cost_evolution.append(cost_only.item() if hasattr(cost_only, "item") else cost_only)
        else:
            self.cost_evolution.append(obj.item() if hasattr(obj, "item") else obj)
        
        self.logger.debug("Outer loop forward pass completed")

    def backward_pass(self, it):
        self.logger.debug("Backward pass started")
        propagationDict = {}
        
        for block_name, block in list(self.blocks.items())[::-1]:
            self.logger.debug("Backward block "+block_name)
            propagationDict = block.backward(propagationDict)
        
        self.logger.debug("Outer loop backward pass completed")

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