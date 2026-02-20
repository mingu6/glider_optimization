from typing import Dict
import logging
import random
import numpy as np
import wandb
import torch
from pathlib import Path
from glider_optimization.config import Config
from glider_optimization.blockBase import Block
from glider_optimization.blocks import Airfoil, NeuralFoilSampling, ReducedModel, OCP, Evaluation


class Runner:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.wandb_enabled = config.io.wandb.enabled
        self.start_iteration = 0
        self._current_iteration = -1
        self._last_completed_iteration = -1
        
        if self.wandb_enabled:
            self._init_wandb()
        
        self.blocks: Dict[str, Block] = {
            "Airfoil": Airfoil(config),
            "NeuralFoilSampling": NeuralFoilSampling(config),
            "ReducedModel": ReducedModel(config),
            "OCP": OCP(config),
            "Evaluation": Evaluation(config)
        }
        
        self._setup_environment()
        if bool(getattr(self.config.run, "continue_run", False)):
            self._maybe_resume_from_checkpoint()

    def _init_wandb(self):
        cfg = self.config
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

    def run(self):
        self.logger.info("Runner started")
        num_iterations = self.config.run.max_outer_iters

        if self.start_iteration > 0:
            self.logger.info(f"Resuming run from iteration {self.start_iteration + 1}/{num_iterations}")

        for iteration in range(self.start_iteration, num_iterations):
            self._current_iteration = iteration
            if iteration % self.config.io.log_every == 0:
                self.logger.info("=" * 100)
                self.logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            self._forward_pass(iteration)
            self._backward_pass(iteration)
            self._last_completed_iteration = iteration
            self.save_checkpoint(next_iteration=iteration + 1, reason="iteration")
        
        self.logger.info("Runner finished")
        self.save_checkpoint(next_iteration=num_iterations, reason="finished")
        if self.wandb_enabled:
            wandb.finish()

    def _checkpoint_path(self) -> Path:
        run_ckpt = getattr(self.config.run, "resume_checkpoint_path", None)
        if run_ckpt:
            return Path(run_ckpt)
        return Path(self.config.io.checkpoint_dir) / "run_state.pt"

    def _maybe_resume_from_checkpoint(self):
        ckpt_path = self._checkpoint_path()
        if not ckpt_path.exists():
            self.logger.warning(f"continue_run=true but checkpoint not found at {ckpt_path}; starting from scratch")
            return

        checkpoint = torch.load(str(ckpt_path), map_location=self.config.run.device)
        self._apply_checkpoint(checkpoint)
        self.start_iteration = int(checkpoint.get("next_iteration", 0))

        max_iters = int(self.config.run.max_outer_iters)
        if self.start_iteration >= max_iters:
            self.logger.warning(
                f"Checkpoint next_iteration={self.start_iteration} >= max_outer_iters={max_iters}; nothing to run"
            )

        self.logger.info(f"Loaded checkpoint from {ckpt_path}")

    def _apply_checkpoint(self, checkpoint: dict):
        airfoil = self.blocks.get("Airfoil", None)
        if airfoil is not None:
            af_state = checkpoint.get("airfoil", {})
            device = airfoil.upper_params.device
            with torch.no_grad():
                if "upper_params" in af_state:
                    airfoil.upper_params.copy_(af_state["upper_params"].to(device))
                if "lower_params" in af_state:
                    airfoil.lower_params.copy_(af_state["lower_params"].to(device))
                if "leading_edge_param" in af_state:
                    airfoil.leading_edge_param.copy_(af_state["leading_edge_param"].to(device))
                if "TE_thickness_param" in af_state:
                    airfoil.TE_thickness_param.copy_(af_state["TE_thickness_param"].to(device))

                if getattr(airfoil, "spanwise_enabled", False):
                    if "upper_params_tip" in af_state:
                        airfoil.upper_params_tip.copy_(af_state["upper_params_tip"].to(device))
                    if "lower_params_tip" in af_state:
                        airfoil.lower_params_tip.copy_(af_state["lower_params_tip"].to(device))
                    if "leading_edge_param_tip" in af_state:
                        airfoil.leading_edge_param_tip.copy_(af_state["leading_edge_param_tip"].to(device))
                    if "TE_thickness_param_tip" in af_state:
                        airfoil.TE_thickness_param_tip.copy_(af_state["TE_thickness_param_tip"].to(device))

            opt_state = af_state.get("optimizer_state", None)
            if opt_state is not None:
                airfoil.optimizer.load_state_dict(opt_state)

            sched_state = af_state.get("scheduler_state", None)
            if sched_state is not None and airfoil.scheduler is not None:
                airfoil.scheduler.load_state_dict(sched_state)

        nf = self.blocks.get("NeuralFoilSampling", None)
        if nf is not None:
            nf_state = checkpoint.get("neuralFoilSampling", {})
            if "lambda_conf" in nf_state:
                nf.lambda_conf = torch.tensor(float(nf_state["lambda_conf"]), dtype=torch.float32, device=nf.device)
            if "lambda_clcd" in nf_state:
                nf.lambda_clcd = torch.tensor(float(nf_state["lambda_clcd"]), dtype=torch.float32, device=nf.device)

        ocp = self.blocks.get("OCP", None)
        if ocp is not None:
            ocp_state = checkpoint.get("ocp", {})
            coc = getattr(ocp, "coc", None)
            if coc is not None:
                coc.w_opt_prev = ocp_state.get("w_opt_prev", None)
                coc.lam_g_prev = ocp_state.get("lam_g_prev", None)
                coc.lam_x_prev = ocp_state.get("lam_x_prev", None)

    def _collect_checkpoint(self, next_iteration: int) -> dict:
        airfoil = self.blocks["Airfoil"]
        airfoil_state = {
            "upper_params": airfoil.upper_params.detach().cpu(),
            "lower_params": airfoil.lower_params.detach().cpu(),
            "leading_edge_param": airfoil.leading_edge_param.detach().cpu(),
            "TE_thickness_param": airfoil.TE_thickness_param.detach().cpu(),
            "optimizer_state": airfoil.optimizer.state_dict(),
            "scheduler_state": airfoil.scheduler.state_dict() if airfoil.scheduler is not None else None,
        }
        if getattr(airfoil, "spanwise_enabled", False):
            airfoil_state.update({
                "upper_params_tip": airfoil.upper_params_tip.detach().cpu(),
                "lower_params_tip": airfoil.lower_params_tip.detach().cpu(),
                "leading_edge_param_tip": airfoil.leading_edge_param_tip.detach().cpu(),
                "TE_thickness_param_tip": airfoil.TE_thickness_param_tip.detach().cpu(),
            })

        nf = self.blocks["NeuralFoilSampling"]
        nf_state = {
            "lambda_conf": float(nf.lambda_conf.detach().cpu().item()) if isinstance(nf.lambda_conf, torch.Tensor) else float(nf.lambda_conf),
            "lambda_clcd": float(nf.lambda_clcd.detach().cpu().item()) if isinstance(nf.lambda_clcd, torch.Tensor) else float(nf.lambda_clcd),
        }

        ocp = self.blocks["OCP"]
        coc = getattr(ocp, "coc", None)
        ocp_state = {
            "w_opt_prev": getattr(coc, "w_opt_prev", None),
            "lam_g_prev": getattr(coc, "lam_g_prev", None),
            "lam_x_prev": getattr(coc, "lam_x_prev", None),
        }

        return {
            "next_iteration": int(next_iteration),
            "airfoil": airfoil_state,
            "neuralFoilSampling": nf_state,
            "ocp": ocp_state,
            "seed": int(self.config.run.seed),
        }

    def save_checkpoint(self, next_iteration: int, reason: str = "manual"):
        ckpt_path = self._checkpoint_path()
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self._collect_checkpoint(next_iteration=next_iteration)
        torch.save(checkpoint, str(ckpt_path))
        self.logger.debug(f"Saved checkpoint ({reason}) to {ckpt_path}")

    def checkpoint_on_interrupt(self):
        next_iter = max(0, self._last_completed_iteration + 1)
        self.save_checkpoint(next_iteration=next_iter, reason="interrupt")
        self.logger.info(f"Checkpoint saved on interrupt at next_iteration={next_iter}")

    def _forward_pass(self, iteration):
        self.logger.debug("Forward pass started")
        
        data = {"iteration": iteration}
        for block_name, block in self.blocks.items():
            self.logger.debug(f"Forward block {block_name}")
            data = block.forward(data)
            # Pass NeuralFoilSampling block reference for 3D LLT artifact export
            if block_name == "NeuralFoilSampling":
                data["_neuralfoil_block_ref"] = block
        
        self.logger.debug("Outer loop forward pass completed")

    def _backward_pass(self, iteration):
        self.logger.debug("Backward pass started")
        
        data = {}
        for block_name, block in reversed(self.blocks.items()):
            self.logger.debug(f"Backward block {block_name}")
            data = block.backward(data)
        
        self.logger.debug("Outer loop backward pass completed")