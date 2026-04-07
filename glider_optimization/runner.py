from typing import Dict
import logging
import random
import numpy as np
import wandb
from glider_optimization.config import Config
from glider_optimization.blockBase import Block
from glider_optimization.blocks import Airfoil, Airfoil3D, NeuralFoilSampling, NeuralFoilSampling3D, ReducedModel, OCP, Evaluation
from .utils.resume import load_checkpoint_from_wandb


class Runner:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.wandb_enabled = config.io.wandb.enabled
        self._resume_from_checkpoint = config.io.wandb.checkpoint_run_id is not None and config.io.wandb.checkpoint_iteration is not None
        self._cost_residual_counter = 0
        self._prev_cost = None
        self._best_cost = float("inf")
        self._best_cost_iter = -1
        self._best_objective = float("inf")
        self._best_objective_iter = -1
        if self.wandb_enabled:
            self._init_wandb()
        
        use_3d = config.neuralFoilSampling.use_3d_llt
        airfoil_block = Airfoil3D(config) if use_3d else Airfoil(config)
        sampling_block = NeuralFoilSampling3D(config) if use_3d else NeuralFoilSampling(config)
        self.blocks: Dict[str, Block] = {
            "Airfoil": airfoil_block,
            "NeuralFoilSampling": sampling_block,
            "ReducedModel": ReducedModel(config),
            "OCP": OCP(config),
            "Evaluation": Evaluation(config)
        }
        
        self.start_iteration = 0
        self._setup_environment()
        
        if self._resume_from_checkpoint:
            self._resume()
            print(f"✓ Initialized state from wandb checkpoint: run={config.io.wandb.checkpoint_run_id}, iter={config.io.wandb.checkpoint_iteration}")

    def _init_wandb(self):
        cfg = self.config
        if self._resume_from_checkpoint:
            wandb.init(
                project=cfg.io.wandb.project,
                entity=cfg.io.wandb.entity,
                name=cfg.io.run_name,
                id=cfg.io.wandb.checkpoint_run_id,
                resume="allow",
                config={
                    "seed": cfg.run.seed,
                    "device": cfg.run.device,
                    "max_outer_iters": cfg.run.max_outer_iters,
                    "airfoil_lr": cfg.airfoil.lr,
                    "neuralfoil_size": cfg.neuralFoilSampling.neuralFoil_size,
                    "n_samples": cfg.neuralFoilSampling.n_samples,
                    "chebyshev_degree": cfg.reducedModel.chebyshev_degree,
                },
                tags=cfg.io.wandb.tags,
                notes=cfg.io.wandb.notes,
            )
        else:
            wandb.init(
                project=cfg.io.wandb.project,
                entity=cfg.io.wandb.entity,
                name=cfg.io.run_name,
                config={
                    "seed": cfg.run.seed,
                    "device": cfg.run.device,
                    "max_outer_iters": cfg.run.max_outer_iters,
                    "airfoil_lr": cfg.airfoil.lr,
                    "neuralfoil_size": cfg.neuralFoilSampling.neuralFoil_size,
                    "n_samples": cfg.neuralFoilSampling.n_samples,
                    "chebyshev_degree": cfg.reducedModel.chebyshev_degree,
                },
                tags=cfg.io.wandb.tags,
                notes=cfg.io.wandb.notes,
            )
        self.logger.info(f"W&B initialized: project={cfg.io.wandb.project}, run={cfg.io.run_name}")

    def _setup_environment(self):
        seed = self.config.run.seed
        np.random.seed(seed)
        random.seed(seed)

        self.logger.info(f"Environment initialized with seed {seed}")
        
    def _resume(self):
        checkpoint_params = load_checkpoint_from_wandb(
            run_id=self.config.io.wandb.checkpoint_run_id,
            iteration=self.config.io.wandb.checkpoint_iteration,
            project=self.config.io.wandb.project,
            entity=self.config.io.wandb.entity
        )
        
        for _, b in self.blocks.items():
            b.resume(checkpoint_params)
        
        self.start_iteration = self.config.io.wandb.checkpoint_iteration + 1

    def run(self):
        self.logger.info("Runner started")
        num_iterations = self.config.run.max_outer_iters

        for iteration in range(self.start_iteration, num_iterations):
            if iteration % self.config.io.log_every == 0:
                self.logger.info("=" * 100)
                self.logger.info(f"Iteration {iteration}/{num_iterations}")
            
            fwd_data = self._forward_pass(iteration)
            self._maybe_save_best_trajectory_snapshots(iteration, fwd_data)

            if self._should_stop_on_cost_target(iteration, fwd_data):
                self.logger.info(f"Cost target reached at iteration {iteration}. Stopping early.")
                break

            self._backward_pass(iteration)

            if self._should_stop_on_cost_residual(iteration, fwd_data):
                self.logger.info(f"Cost residual criterion reached at iteration {iteration}. Stopping early.")
                break
        
        self._plot_objective_if_needed()
        self.logger.info("Runner finished")
        if self.wandb_enabled:
            wandb.finish()

    def _forward_pass(self, iteration):
        self.logger.debug("Forward pass started")
        
        data = {"iteration": iteration}
        for block_name, block in self.blocks.items():
            self.logger.debug(f"Forward block {block_name}")
            data = block.forward(data)
        
        self.logger.debug("Outer loop forward pass completed")
        return data

    def _backward_pass(self, iteration):
        self.logger.debug("Backward pass started")
        
        data = {}
        for block_name, block in reversed(self.blocks.items()):
            self.logger.debug(f"Backward block {block_name}")
            data = block.backward(data)
        
        self.logger.debug("Outer loop backward pass completed")

    def _should_stop_on_cost_residual(self, iteration: int, fwd_data: dict) -> bool:
        tol = getattr(self.config.run, "cost_residual_tol", None)
        if tol is None:
            return False

        if "cost" not in fwd_data:
            return False

        current_cost = float(fwd_data["cost"])
        if self._prev_cost is None:
            self._prev_cost = current_cost
            return False

        residual = abs(current_cost - self._prev_cost)
        self._prev_cost = current_cost

        min_iters = max(0, int(getattr(self.config.run, "cost_residual_min_iters", 0)))
        patience = max(1, int(getattr(self.config.run, "cost_residual_patience", 1)))

        if residual < float(tol) and iteration >= min_iters:
            self._cost_residual_counter += 1
        else:
            self._cost_residual_counter = 0

        if iteration % self.config.io.log_every == 0:
            self.logger.info(
                f"Cost residual={residual:.6e}, tol={float(tol):.6e}, "
                f"patience_counter={self._cost_residual_counter}/{patience}"
            )

        return self._cost_residual_counter >= patience

    def _should_stop_on_cost_target(self, iteration: int, fwd_data: dict) -> bool:
        target = getattr(self.config.run, "cost_target", None)
        if target is None:
            return False
        if "cost" not in fwd_data:
            return False

        min_iters = max(0, int(getattr(self.config.run, "cost_target_min_iters", 0)))
        if iteration < min_iters:
            return False

        current_cost = float(fwd_data["cost"])
        if iteration % self.config.io.log_every == 0:
            self.logger.info(f"Cost target check: cost={current_cost:.6f}, target={float(target):.6f}")

        return current_cost <= float(target)

    def _plot_objective_if_needed(self):
        eval_block = self.blocks.get("Evaluation")
        if eval_block is not None and not self.wandb_enabled:
            try:
                eval_block.plot_objective()
            except Exception as exc:
                self.logger.warning(f"Failed to save objective plots: {exc}")

    def _maybe_save_best_trajectory_snapshots(self, iteration: int, fwd_data: dict) -> None:
        ocp_block = self.blocks.get("OCP")
        if ocp_block is None:
            return
        if not hasattr(ocp_block, "save_best_snapshot"):
            return

        current_cost = fwd_data.get("cost")
        if current_cost is not None:
            current_cost = float(current_cost)
            if current_cost < self._best_cost:
                self._best_cost = current_cost
                self._best_cost_iter = iteration
                ocp_block.save_best_snapshot(
                    metric_name="Cost",
                    metric_value=self._best_cost,
                    best_iteration=self._best_cost_iter,
                    filename_suffix="best_cost",
                )

        current_obj = fwd_data.get("total_obj")
        if current_obj is not None:
            current_obj = float(current_obj)
            if current_obj < self._best_objective:
                self._best_objective = current_obj
                self._best_objective_iter = iteration
                ocp_block.save_best_snapshot(
                    metric_name="Objective",
                    metric_value=self._best_objective,
                    best_iteration=self._best_objective_iter,
                    filename_suffix="best_objective",
                )

    def checkpoint_on_interrupt(self):
        self._plot_objective_if_needed()