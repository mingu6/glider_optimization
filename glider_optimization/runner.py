from typing import Dict
import logging
import random
import numpy as np
import os
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
        if self.wandb_enabled:
            self._init_wandb()
        
        use_3d = config.neuralFoilSampling.use_3d_llt
        self.blocks: Dict[str, Block] = {
            "Airfoil": Airfoil3D(config) if use_3d else Airfoil(config),
            "NeuralFoilSampling": NeuralFoilSampling3D(config) if use_3d else NeuralFoilSampling(config),
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

    def _chain_state_load(self):
        """Restore Adam moments, LR schedule position and AL duals from a previous link.

        Continuation links are adjacent problems: lambda for the epsilon constraint is
        close to its neighbour's, and Adam's moments describe the same design space.
        Restarting both each link made every link spend its early iterations
        rediscovering what the previous one had already established.
        """
        path = self.config.run.chain_state_path
        if not path or not os.path.exists(path):
            return
        try:
            import torch
            state = torch.load(path, weights_only=False)
        except Exception as exc:
            self.logger.warning(f"Could not read chain state: {exc}")
            return

        af = self.blocks.get("Airfoil")
        if af is not None and "optimizer" in state:
            try:
                af.optimizer.load_state_dict(state["optimizer"])
                if af.scheduler is not None and "scheduler" in state:
                    af.scheduler.load_state_dict(state["scheduler"])
            except Exception as exc:
                self.logger.warning(f"Could not restore optimizer state: {exc}")

        ev = self.blocks.get("Evaluation")
        if ev is not None and "task_al" in state:
            ev.task_al.load_state_dict(state["task_al"])
        nf = self.blocks.get("NeuralFoilSampling") or self.blocks.get("ReducedModel")
        if nf is not None and hasattr(nf, "augmented_lagrangian") and "design_al" in state:
            nf.augmented_lagrangian.load_state_dict(state["design_al"])
        self.logger.info(f"Restored chain state from {path}")

    def _chain_state_save(self):
        path = self.config.run.chain_state_path
        if not path:
            return
        try:
            import torch
            state = {}
            af = self.blocks.get("Airfoil")
            if af is not None:
                state["optimizer"] = af.optimizer.state_dict()
                if af.scheduler is not None:
                    state["scheduler"] = af.scheduler.state_dict()
            ev = self.blocks.get("Evaluation")
            if ev is not None:
                state["task_al"] = ev.task_al.state_dict()
            nf = self.blocks.get("NeuralFoilSampling") or self.blocks.get("ReducedModel")
            if nf is not None and hasattr(nf, "augmented_lagrangian"):
                state["design_al"] = nf.augmented_lagrangian.state_dict()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(state, path)
        except Exception as exc:
            self.logger.warning(f"Could not persist chain state: {exc}")

    def run(self):
        self.logger.info("Runner started")
        self._chain_state_load()
        num_iterations = self.config.run.max_outer_iters

        for iteration in range(self.start_iteration, num_iterations):
            if iteration % self.config.io.log_every == 0:
                self.logger.info("=" * 100)
                self.logger.info(f"Iteration {iteration}/{num_iterations}")
            
            self._forward_pass(iteration)
            self._backward_pass(iteration)
            # Persist periodically, not only at the end: a link that is interrupted
            # would otherwise lose its Adam moments and duals entirely.
            if iteration % max(self.config.io.log_every, 1) == 0:
                self._chain_state_save()
        
        self._chain_state_save()
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