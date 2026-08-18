from typing import Dict
import logging
import random
import numpy as np
import wandb
from glider_optimization.config import Config, EvaluationMode
from glider_optimization.blockBase import Block
from glider_optimization.blocks import Airfoil, Airfoil3D, NeuralFoilSampling, NeuralFoilSampling3D, ReducedModel, OCP, Evaluation, RoboticArm
from .utils.resume import load_checkpoint_from_wandb


class Runner:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.wandb_enabled = config.io.wandb.enabled
        self._resume_from_checkpoint = config.io.wandb.checkpoint_run_id is not None and config.io.wandb.checkpoint_iteration is not None
        if self.wandb_enabled:
            self._init_wandb()
        
        use_3d = config.neuralFoilSampling.use_3d_llt

        if config.evaluation.mode != EvaluationMode.RobotThrowing:
            self.blocks: Dict[str, Block] = {
                "Airfoil": Airfoil3D(config) if use_3d else Airfoil(config),
                "NeuralFoilSampling": NeuralFoilSampling3D(config) if use_3d else NeuralFoilSampling(config),
                "ReducedModel": ReducedModel(config),
                "OCP": OCP(config),
                "Evaluation": Evaluation(config)
            }
        else:
            self.blocks: Dict[str, Block] = {
                "Robot": RoboticArm(config),
                "OCP": OCP(config),
                "Evaluation": Evaluation(config)
            }
        
        self.start_iteration = 0
        self._setup_environment()
        
        if self._resume_from_checkpoint:
            self._resume()
            print(f"✓ Initialized state from wandb checkpoint: run={config.io.wandb.checkpoint_run_id}, iter={config.io.wandb.checkpoint_iteration}")

    def _wandb_config(self):
        cfg = self.config
        common = {
            "seed": cfg.run.seed,
            "device": cfg.run.device,
            "max_outer_iters": cfg.run.max_outer_iters,
            "evaluation_mode": cfg.evaluation.mode.value,
        }
        if cfg.evaluation.mode == EvaluationMode.RobotThrowing:
            return common | {
                "arm_lr": cfg.arm.lr,
                "horizon": cfg.arm.horizon,
                "integrator": cfg.arm.integrator,
                "targets": cfg.arm.targets,
                "struct_mass": cfg.arm.struct_mass,
                "torque_budget": cfg.arm.torque_budget,
            }
        return common | {
            "airfoil_lr": cfg.airfoil.lr,
            "neuralfoil_size": cfg.neuralFoilSampling.neuralFoil_size,
            "n_samples": cfg.neuralFoilSampling.n_samples,
            "chebyshev_degree": cfg.reducedModel.chebyshev_degree,
        }

    def _init_wandb(self):
        cfg = self.config
        if self._resume_from_checkpoint:
            wandb.init(
                project=cfg.io.wandb.project,
                entity=cfg.io.wandb.entity,
                name=cfg.io.run_name,
                id=cfg.io.wandb.checkpoint_run_id,
                resume="allow",
                config=self._wandb_config(),
                tags=cfg.io.wandb.tags,
                notes=cfg.io.wandb.notes,
            )
        else:
            wandb.init(
                project=cfg.io.wandb.project,
                entity=cfg.io.wandb.entity,
                name=cfg.io.run_name,
                config=self._wandb_config(),
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

    def _backward_pass(self, iteration):
        self.logger.debug("Backward pass started")
        
        data = {}
        for block_name, block in reversed(self.blocks.items()):
            self.logger.debug(f"Backward block {block_name}")
            data = block.backward(data)
        
        self.logger.debug("Outer loop backward pass completed")