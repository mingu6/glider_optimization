from typing import Any, Dict
from .config import Config
from .blockBase import Block
from .blocks import Airfoil, NeuralFoilSampling, ReducedModel, Evaluation
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
            "Evaluation": Evaluation(config)
        }
        self.objective_evolution = []
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
        
        obj = propagationDict["objective"]
        if it % self.config.io.log_every == 0:
            self.logger.info(f"Objective = {obj}")
        self.objective_evolution.append(obj.item() if hasattr(obj, "item") else obj)
        
        self.logger.debug("Outer loop forward pass completed")

    def backward_pass(self, it):
        self.logger.debug("Backward pass started")
        propagationDict = {}
        
        for block_name, block in list(self.blocks.items())[::-1]:
            self.logger.debug("Backward block "+block_name)
            propagationDict = block.backward(propagationDict)
        
        self.logger.debug("Outer loop backward pass completed")

    def plot_objective(self):
        matplotlib.use("TkAgg")
        plt.figure()
        plt.plot(self.objective_evolution)
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("Optimization Progress")
        plt.grid(True)
        plt.show()