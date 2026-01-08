from ..blockBase import Block
from typing import Dict, Any, override
from ..utils.go_safe_pdp import COCsys
from ..utils.glider_jinenv import GliderPerching
from ..utils.idoc_ineq import build_blocks_idoc, idoc_full
from ..config import Config
from casadi import pi, vertcat
import numpy as np
import torch
class OCP(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.run.device)
        
        self.env = GliderPerching(self.config)
        self.coc = COCsys()
        
        self.env.initDyn()
        env_dyn = self.env.X + self.env.X[-1] * self.env.f
        
        self.env.initCost(state_weights=[10., 10., 3., 0.1, 2., 1., 1., 0.1], wu=0.1)
        self.env.initConstraints(-pi/3, pi/8, 13)
        
        self.coc.setAuxvarVariable(vertcat(self.env.dyn_auxvar))
        self.coc.setStateVariable(self.env.X)
        self.coc.setControlVariable(self.env.U)
        self.coc.setDyn(env_dyn)

        self.coc.setPathCost(self.env.path_cost)
        self.coc.setFinalCost(self.env.final_cost)

        self.coc.setPathInequCstr(self.env.path_inequ)
        self.coc.diffCPMP()
        
        self.init_state = [-3.5, 0.1 , 0. , 0., 7., 0. , 0., 0.01]
        
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        
        weights_CL = downstream_info["phi_CL"].view(-1, 1).detach().cpu().numpy()
        weights_CD = downstream_info["phi_CD"].view(-1, 1).detach().cpu().numpy()
        weights_CM = downstream_info["phi_CM"].view(-1, 1).detach().cpu().numpy()

        auxvar_vector = np.vstack([weights_CL, weights_CD, weights_CM])
        self.last_traj_COC = self.coc.ocSolver(horizon=111, init_state=self.init_state, auxvar_value=auxvar_vector, timeVarying=True)
        
        #self.env.play_animation(traj_COC['state_traj_opt'], traj_COC['control_traj_opt'], save_option=True, title="meek")
        return {
            "objective": self.last_traj_COC["cost"][0][0]
        }
    
    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        delta = 0.00001
        auxsys_COC = self.coc.getAuxSys(opt_sol=self.last_traj_COC, threshold=1e-5)
        idoc_ctx = build_blocks_idoc(auxsys_COC, delta)
        traj_deriv_COC = idoc_full(idoc_ctx)
        
        state_traj = self.last_traj_COC['state_traj_opt']
        dJdeps_traj = self.env.dfinal_cost_dx_fn(state_traj[-1])
        depsdphi_traj = traj_deriv_COC['state_traj_opt']
        dJ_dphi_np = (dJdeps_traj.T @ depsdphi_traj[-1]).T
        dJ_dphi = torch.from_numpy(dJ_dphi_np.full()).float().to(self.device)
        dJ_dphi = dJ_dphi.view(3, -1)

        return {"dJ_dphi": dJ_dphi}
                