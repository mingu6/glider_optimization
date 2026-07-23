from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import wandb

from deap import base, creator, tools, algorithms

from glider_optimization.config import Config, load_config
from glider_optimization.logger import setup_logging
from glider_optimization.blockBase import Block
from glider_optimization.blocks import Airfoil, NeuralFoilSampling, ReducedModel, OCP, Evaluation


if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


class BaselineRunner:

    POP_SIZE: int = 20
    N_GEN: int = 100          # treated as max_outer_iters if that is smaller
    CXPB: float = 0.7         # crossover probability
    MUTPB: float = 0.3        # mutation probability
    ETA_CX: float = 20.0      # SBX eta
    ETA_MUT: float = 20.0     # polynomial mutation eta
    SIGMA: float = 0.1      # Gaussian mutation std (fraction of param range)
    TOURNSIZE: int = 3

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.wandb_enabled = config.io.wandb.enabled

        if self.wandb_enabled:
            self._init_wandb()

        self.airfoil_block = Airfoil(config)
        self.blocks: Dict[str, Block] = {
            "NeuralFoilSampling": NeuralFoilSampling(config),
            "ReducedModel": ReducedModel(config),
            "OCP": OCP(config),
            "Evaluation": Evaluation(config),
        }

        self._setup_environment()
        self._setup_deap()

    def _init_wandb(self):
        cfg = self.config
        wandb.init(
            project=cfg.io.wandb.project,
            entity=cfg.io.wandb.entity,
            name=cfg.io.run_name,
            config={
                "seed": cfg.run.seed,
                "device": cfg.run.device,
                "pop_size": self.POP_SIZE,
                "n_gen": self.N_GEN,
                "cxpb": self.CXPB,
                "mutpb": self.MUTPB,
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

    def _setup_deap(self):
        """Configure the DEAP toolbox."""
        af_cfg = self.config.airfoil
        toolbox = base.Toolbox()

        low: List[float] = (
            [-0.5] * 8   # upper_weights lower bound
            + [-0.5] * 8  # lower_weights lower bound
            + [-0.5]      # leading_edge_weight
            + [0]      # TE_thickness
        )
        high: List[float] = (
            [0.5] * 8
            + [0.5] * 8
            + [0.5]
            + [0.01]
        )
        self._low = low
        self._high = high
        ndim = len(low)

        def _random_individual():
            seed_genome = (
                list(self.config.airfoil.upper_initial_weights) +
                list(self.config.airfoil.lower_initial_weights) +
                [self.config.airfoil.leading_edge_weight] +
                [self.config.airfoil.TE_thickness]
            )
            
            sigma = 0.05

            genome = [
                np.clip(g + random.gauss(0, sigma * (hi - lo)), lo, hi)
                for g, lo, hi in zip(seed_genome, low, high)
            ]
            return creator.Individual(genome)

        toolbox.register("individual", _random_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._evaluate)
        toolbox.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=low, up=high, eta=self.ETA_CX,
        )
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=low, up=high, eta=self.ETA_MUT, indpb=1.0 / ndim,
        )
        toolbox.register("select", tools.selTournament, tournsize=self.TOURNSIZE)

        self.toolbox = toolbox

    def _genome_to_airfoil_params(self, genome: List[float]):
        """Inject a flat genome into the Airfoil block's parameter tensors."""
        upper = torch.tensor(genome[0:8], dtype=torch.float32)
        lower = torch.tensor(genome[8:16], dtype=torch.float32)
        le = torch.tensor([genome[16]], dtype=torch.float32)
        te = torch.tensor([genome[17]], dtype=torch.float32)
        
        te = te.clamp(1e-4, 0.01)

        min_gap = 0.05
        upper = torch.maximum(upper, lower + min_gap)

        self.airfoil_block.upper_params = nn.Parameter(upper)
        self.airfoil_block.lower_params = nn.Parameter(lower)
        self.airfoil_block.leading_edge_param = nn.Parameter(le)
        self.airfoil_block.TE_thickness_param = nn.Parameter(te)

    def _evaluate(self, individual: List[float]) -> Tuple[float]:
        """
        DEAP evaluation function.

        Returns a 1-tuple (total_objective,) – DEAP requires a tuple.
        No backward / gradient computation is performed.
        """
        self._genome_to_airfoil_params(individual)

        # Forward pass through the Airfoil block (no logging, no plotting)
        data: Dict[str, Any] = {"iteration": self._current_gen}
        data = self.airfoil_block.forward(data)

        # Forward pass through the remaining blocks
        for block in self.blocks.values():
            data = block.forward(data)

        return (data["total_obj"],)

    def run(self):
        self.logger.info("BaselineRunner (DEAP) started")

        n_gen = min(self.N_GEN, self.config.run.max_outer_iters)
        pop = self.toolbox.population(n=self.POP_SIZE)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("mean", np.mean)
        stats.register("max", np.max)

        self._current_gen = 1

        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop)
        self.logger.info(f"Gen 0 | {record}")
        self._log_stats(0, record, hof[0].fitness.values[0])

        for gen in range(1, n_gen + 1):
            self._current_gen = gen

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)
            record = stats.compile(pop)

            if gen % self.config.io.log_every == 0:
                self.logger.info(f"Gen {gen}/{n_gen} | {record}")

            self._log_stats(gen, record, hof[0].fitness.values[0])

        best = hof[0]
        self.logger.info(
            f"BaselineRunner finished. Best objective = {best.fitness.values[0]:.6f}"
        )
        self.logger.info(f"Best genome: {best}")

        if self.wandb_enabled:
            wandb.finish()

    def _log_stats(self, gen: int, record: dict, best_obj: float):
        if self.wandb_enabled:
            wandb.log(
                {
                    "baseline/generation": gen,
                    "baseline/best_objective": best_obj,
                    "baseline/mean_objective": record["mean"],
                    "baseline/min_objective": record["min"],
                    "baseline/max_objective": record["max"],
                },
                step=gen,
            )


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="genetic-baseline")
    default_config = Path(__file__).resolve().parents[1] / "conf" / "perching_2D.yaml"
    p.add_argument("--config", "-c", type=Path, default=default_config)
    p.add_argument("--run-name", "-n", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    return p


def parse_args(args: Optional[list] = None):
    return build_parser().parse_args(args)


def _apply_overrides(cfg: Config, args) -> Config:
    if args.device is not None:
        cfg.run.device = args.device
    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.run_name is not None:
        cfg.io.run_name = args.run_name
    cfg.io.debug = bool(args.debug)
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    config = _apply_overrides(config, args)
    setup_logging(config.io)

    runner = BaselineRunner(config)
    runner.run()
