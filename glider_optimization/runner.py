from typing import Dict
from glider_optimization.config import Config
from glider_optimization.blockBase import Block
from glider_optimization.blocks import Airfoil, NeuralFoilSampling, ReducedModel, OCP, Evaluation
import logging
import wandb
class Runner:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        
        self.wandb_enabled = config.io.wandb.enabled
        if self.wandb_enabled:
            wandb.init(
                project=config.io.wandb.project,
                entity=config.io.wandb.entity,
                name=config.io.run_name,
                config={
                    "seed": config.run.seed,
                    "device": config.run.device,
                    "max_outer_iters": config.run.max_outer_iters,
                    "airfoil_lr": config.airfoil.lr,
                    "neuralfoil_size": config.neuralFoilSampling.neuralFoil_size,
                    "n_samples": config.neuralFoilSampling.n_samples,
                    "chebyshev_degree": config.reducedModel.chebyshev_degree,
                },
                tags=config.io.wandb.tags,
                notes=config.io.wandb.notes,
            )
            self.logger.info(f"W&B initialized: project={config.io.wandb.project}, run={config.io.run_name}")
        
        
        # The order matters! 
        # The output of block i-1 is the input of block i in the forward phase
        # The output of block i is the input of block i-1 in the backward phase
        self.blocks : Dict[str, Block] = {
            "Airfoil": Airfoil(config),
            "NeuralFoilSampling": NeuralFoilSampling(config),
            "ReducedModel": ReducedModel(config),
            "OCP": OCP(config),
            "Evaluation": Evaluation(config)
        }

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
        
        self.logger.info("Runner finished")
        if self.wandb_enabled:
            wandb.finish()
        
    def checkpoint_on_interrupt(): ... # TODO

    def forward_pass(self, it):
        self.logger.debug("Forward pass started")
        
        propagationDict = {"iteration": it}
        for block_name, block in self.blocks.items():
            self.logger.debug("Forward block "+block_name)
            propagationDict = block.forward(propagationDict)
                
        self.logger.debug("Outer loop forward pass completed")

    def backward_pass(self, it):
        self.logger.debug("Backward pass started")
        propagationDict = {}
        
        for block_name, block in list(self.blocks.items())[::-1]:
            self.logger.debug("Backward block "+block_name)
            propagationDict = block.backward(propagationDict)
        
        self.logger.debug("Outer loop backward pass completed")