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
            
            self._forward_pass(iteration)
            self._backward_pass(iteration)
        
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

    def checkpoint_on_interrupt(self):
        pass