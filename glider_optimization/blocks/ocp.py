from pathlib import Path
from ..blockBase import Block
from typing import Dict, Any, override
from ..utils.go_safe_pdp import COCsys
from ..utils.glider_jinenv import GliderPerching
from ..utils.idoc_ineq import build_blocks_idoc, idoc_full
from ..config import Config
from casadi import pi, vertcat
import numpy as np
import torch
import logging
import wandb
import tempfile

class OCP(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.run.device)
        self.logger = logging
        
        self.env = GliderPerching(self.config)
        self.coc = COCsys()
        
        self.env.initDyn()
        env_dyn = self.env.X + self.env.X[-1] * self.env.f
        
        self.env.initCost(state_weights=config.ocp.terminal_state_weight, wu=config.ocp.stage_control_weight)
        self.env.initConstraints(-pi/3, pi/8, 13)
        
        self.coc.setAuxvarVariable(vertcat(self.env.dyn_auxvar))
        self.coc.setStateVariable(self.env.X)
        self.coc.setControlVariable(self.env.U)
        self.coc.setDyn(env_dyn)

        self.coc.setPathCost(self.env.path_cost)
        self.coc.setFinalCost(self.env.final_cost)

        self.coc.setPathInequCstr(self.env.path_inequ)
        self.coc.diffCPMP()
        
        self.init_state = config.ocp.initial_state
        
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        
        weights_CL = downstream_info["phi_CL"].view(-1, 1).detach().cpu().numpy()
        weights_CD = downstream_info["phi_CD"].view(-1, 1).detach().cpu().numpy()
        weights_CM = downstream_info["phi_CM"].view(-1, 1).detach().cpu().numpy()

        # Wing always present
        aux_blocks = [weights_CL, weights_CD, weights_CM]

        # Optional elevator coefficients (in 3D mode with fixed-elevator LLT)
        if "phi_CL_e" in downstream_info and "phi_CD_e" in downstream_info and "phi_CM_e" in downstream_info:
            aux_blocks += [
                downstream_info["phi_CL_e"].view(-1, 1).detach().cpu().numpy(),
                downstream_info["phi_CD_e"].view(-1, 1).detach().cpu().numpy(),
                downstream_info["phi_CM_e"].view(-1, 1).detach().cpu().numpy(),
            ]

        auxvar_vector = np.vstack(aux_blocks)

        #auxvar_vector = np.vstack([weights_CL, weights_CD, weights_CM])
        self.last_traj_COC = self.coc.ocSolver(horizon=111, init_state=self.init_state, auxvar_value=auxvar_vector, timeVarying=True, warm_start=True)

        if not self.last_traj_COC["success"]:
            self.logger.critical(f"⚠️ IPOPT couldn't find a solution")
            
        num_iterations = self.config.run.max_outer_iters
        iteration = downstream_info["iteration"]
        if iteration == 0 or iteration == (num_iterations - 1):
            try: 
                self.plot(iteration)
            except ValueError:
                pass

        return {
            "trajectory": self.last_traj_COC,
            "iteration": downstream_info["iteration"]
        }
    
    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        delta = 0.00001
        auxsys_COC = self.coc.getAuxSys(opt_sol=self.last_traj_COC, threshold=1e-5)
        idoc_ctx = build_blocks_idoc(auxsys_COC, delta)
        traj_deriv_COC = idoc_full(idoc_ctx)

        dJ_deps = upstream_grads["dJ_deps"]
        
        deps_dphi = traj_deriv_COC['state_traj_opt'][-1]
        
        dJ_dphi_np = deps_dphi.T @ dJ_deps
        dJ_dphi = torch.from_numpy(dJ_dphi_np).float().to(self.device)
        #dJ_dphi = dJ_dphi.view(3, -1)

        # Number of Chebyshev coefficients per surrogate block - Before only wing CL, CD, CM, now possibly also elevator CL_e, CD_e, CM_e
        cheb_deg = self.config.reducedModel.chebyshev_degree
        n_feat = (cheb_deg + 1) ** 2
        n_blocks = int(dJ_dphi.numel() // n_feat)
        dJ_dphi = dJ_dphi.view(n_blocks, -1)
        
        return {"dJ_dphi": dJ_dphi}
                
    def plot(self, iteration):
        run_name = getattr(self.config.io, "run_name", "run")
        traj = self.last_traj_COC

        if self.config.io.wandb.enabled:
            with tempfile.TemporaryDirectory() as tmpdir:
                title = Path(tmpdir) / f"{run_name}_traj_iter{iteration}"
                self.env.play_animation(
                    traj['state_traj_opt'],
                    traj['control_traj_opt'],
                    save_option=True,
                    title=str(title),
                    fps=self.config.io.gif_fps
                )
                gif_path = f"{title}.gif"
                wandb.log(
                    {f"trajectory/traj_iter_{iteration}": wandb.Video(gif_path, format="gif")},
                    step=iteration
                )
        else:
            out_dir = Path(self.config.io.checkpoint_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            title = out_dir / f"{run_name}_traj_iter{iteration}"
            self.env.play_animation(
                traj['state_traj_opt'],
                traj['control_traj_opt'],
                save_option=True,
                title=str(title),
                fps=self.config.io.gif_fps
            )