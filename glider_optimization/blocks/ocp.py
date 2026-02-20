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
import csv

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

    def _log_rollout_term_breakdown(self, auxvar_vector: np.ndarray, start_stage: int = 10, end_stage: int = 14) -> None:
        """Log term-by-term dynamics quantities over a short rollout window."""
        try:
            xk = np.array(self.init_state, dtype=float)
            u0 = [0.5 * (lb + ub) for lb, ub in zip(self.coc.control_lb, self.coc.control_ub)]

            names = getattr(self.env, "dyn_term_names", None)
            if not names or not hasattr(self.env, "dyn_terms_fn"):
                self.logger.warning("🔍 Term breakdown unavailable: dyn_terms_fn not initialized")
                return

            idx = {n: i for i, n in enumerate(names)}

            for stage in range(1, end_stage + 1):
                terms = np.array(self.env.dyn_terms_fn(xk.tolist(), u0, auxvar_vector)).reshape(-1)
                xnext = np.array(self.coc.dyn_fn(xk.tolist(), u0, auxvar_vector)).reshape(-1)

                if start_stage <= stage <= end_stage:
                    self.logger.info(
                        "🔍 STAGE %d pre-step | max|x|=%.3e theta=%.3e thetadot=%.3e xdot=%.3e zdot=%.3e",
                        stage, np.max(np.abs(xk)), xk[2], xk[6], xk[4], xk[5]
                    )
                    self.logger.info(
                        "🔍 STAGE %d aero wing: v=%.3e alpha=%.3e Re=%.3e CL=%.3e CD=%.3e CM=%.3e",
                        stage,
                        terms[idx["v_w_safe"]], terms[idx["alpha_w"]], terms[idx["Re"]],
                        terms[idx["CL_w"]], terms[idx["CD_w"]], terms[idx["CM_w"]],
                    )
                    self.logger.info(
                        "🔍 STAGE %d aero elev: v=%.3e alpha=%.3e Re=%.3e CL=%.3e CD=%.3e CM=%.3e",
                        stage,
                        terms[idx["v_e_safe"]], terms[idx["alpha_e"]], terms[idx["Re_e"]],
                        terms[idx["CL_e"]], terms[idx["CD_e"]], terms[idx["CM_e"]],
                    )
                    self.logger.info(
                        "🔍 STAGE %d forces/moments: Fw=(%.3e, %.3e) Fe=(%.3e, %.3e) Mw=%.3e Me=%.3e tau_w=%.3e tau_e=%.3e",
                        stage,
                        terms[idx["F_wx"]], terms[idx["F_wz"]],
                        terms[idx["F_ex"]], terms[idx["F_ez"]],
                        terms[idx["M_w"]], terms[idx["M_e"]],
                        terms[idx["tau_w"]], terms[idx["tau_e"]],
                    )
                    self.logger.info(
                        "🔍 STAGE %d tau_e terms: r_x*F_z=%.3e  -r_z*F_x=%.3e  M_e=%.3e",
                        stage,
                        terms[idx["tau_e_term_rxfz"]],
                        terms[idx["tau_e_term_rzfx"]],
                        terms[idx["tau_e_term_M"]],
                    )
                    self.logger.info(
                        "🔍 STAGE %d accelerations: xddot=%.3e zddot=%.3e thetaddot=%.3e",
                        stage, terms[idx["xddot"]], terms[idx["zddot"]], terms[idx["thetaddot"]]
                    )
                    self.logger.info(
                        "🔍 STAGE %d post-step | max|x_next|=%.3e theta_next=%.3e thetadot_next=%.3e",
                        stage, np.max(np.abs(xnext)), xnext[2], xnext[6]
                    )

                xk = xnext

        except Exception as exc:
            self.logger.warning(f"🔍 Term breakdown failed: {exc}")

    def _export_rollout_debug_csv(self, auxvar_vector: np.ndarray, max_stage: int = 16) -> None:
        """Export stage-wise rollout terms from stage 0..max_stage for debugging."""
        try:
            out_path = Path("diagnostics/2026-02-13_3d-llt-debug/rollout_terms_stage_0_16.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            xk = np.array(self.init_state, dtype=float)
            u0 = [0.5 * (lb + ub) for lb, ub in zip(self.coc.control_lb, self.coc.control_ub)]
            names = getattr(self.env, "dyn_term_names", None)
            if not names or not hasattr(self.env, "dyn_terms_fn"):
                self.logger.warning("🔍 CSV export unavailable: dyn_terms_fn not initialized")
                return
            idx = {n: i for i, n in enumerate(names)}

            columns = [
                "stage", "x", "z", "theta", "phi", "xdot", "zdot", "thetadot",
                "r_e_x", "r_e_z", "x_edot", "z_edot", "v_e_safe", "q_e",
                "CL_e", "CD_e", "F_ex", "F_ez",
                "tau_e_term_rxfz", "tau_e_term_rzfx", "tau_e_term_M", "tau_e", "thetaddot",
            ]

            rows = []
            for stage in range(0, max_stage + 1):
                terms = np.array(self.env.dyn_terms_fn(xk.tolist(), u0, auxvar_vector)).reshape(-1)
                row = {
                    "stage": stage,
                    "x": float(xk[0]),
                    "z": float(xk[1]),
                    "theta": float(xk[2]),
                    "phi": float(xk[3]),
                    "xdot": float(xk[4]),
                    "zdot": float(xk[5]),
                    "thetadot": float(xk[6]),
                    "r_e_x": float(terms[idx["r_e_x"]]),
                    "r_e_z": float(terms[idx["r_e_z"]]),
                    "x_edot": float(terms[idx["x_edot"]]),
                    "z_edot": float(terms[idx["z_edot"]]),
                    "v_e_safe": float(terms[idx["v_e_safe"]]),
                    "q_e": float(terms[idx["q_e"]]),
                    "CL_e": float(terms[idx["CL_e"]]),
                    "CD_e": float(terms[idx["CD_e"]]),
                    "F_ex": float(terms[idx["F_ex"]]),
                    "F_ez": float(terms[idx["F_ez"]]),
                    "tau_e_term_rxfz": float(terms[idx["tau_e_term_rxfz"]]),
                    "tau_e_term_rzfx": float(terms[idx["tau_e_term_rzfx"]]),
                    "tau_e_term_M": float(terms[idx["tau_e_term_M"]]),
                    "tau_e": float(terms[idx["tau_e"]]),
                    "thetaddot": float(terms[idx["thetaddot"]]),
                }
                rows.append(row)
                if stage < max_stage:
                    xk = np.array(self.coc.dyn_fn(xk.tolist(), u0, auxvar_vector)).reshape(-1)

            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)

            self.logger.info(f"🔍 Exported rollout term CSV to {out_path}")
        except Exception as exc:
            self.logger.warning(f"🔍 CSV export failed: {exc}")

    def _export_rollout_debug_full_csv(self, auxvar_vector: np.ndarray, max_stage: int = 16) -> None:
        """Export full stage-wise rollout terms (states + all dyn_terms) without overwriting baseline CSV."""
        try:
            out_path = Path("diagnostics/2026-02-13_3d-llt-debug/rollout_terms_stage_0_16_full.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            xk = np.array(self.init_state, dtype=float)
            u0 = [0.5 * (lb + ub) for lb, ub in zip(self.coc.control_lb, self.coc.control_ub)]
            term_names = getattr(self.env, "dyn_term_names", None)
            if not term_names or not hasattr(self.env, "dyn_terms_fn"):
                self.logger.warning("🔍 Full CSV export unavailable: dyn_terms_fn not initialized")
                return

            state_names = ["x", "z", "theta", "phi", "xdot", "zdot", "thetadot", "t"]
            columns = ["stage"] + state_names + term_names

            rows = []
            for stage in range(0, max_stage + 1):
                terms = np.array(self.env.dyn_terms_fn(xk.tolist(), u0, auxvar_vector)).reshape(-1)

                row = {"stage": int(stage)}
                for i, name in enumerate(state_names):
                    row[name] = float(xk[i])
                for i, name in enumerate(term_names):
                    row[name] = float(terms[i])

                rows.append(row)

                if stage < max_stage:
                    xk = np.array(self.coc.dyn_fn(xk.tolist(), u0, auxvar_vector)).reshape(-1)

            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)

            self.logger.info(f"🔍 Exported FULL rollout term CSV to {out_path}")
        except Exception as exc:
            self.logger.warning(f"🔍 Full CSV export failed: {exc}")
        
        
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

        # DIAGNOSTIC: Log auxvar values going into OCP
        self.logger.info(f"🔍 OCP auxvar_vector shape: {auxvar_vector.shape}")
        self.logger.info(f"🔍 OCP auxvar stats - min={np.min(auxvar_vector):.6f}, max={np.max(auxvar_vector):.6f}, mean={np.mean(auxvar_vector):.6f}")
        has_nan = np.isnan(auxvar_vector).any()
        has_inf = np.isinf(auxvar_vector).any()
        if has_nan or has_inf:
            self.logger.error(f"🚨 OCP auxvar contains NaN={has_nan}, Inf={has_inf}")
            self.logger.error(f"🚨 NaN count: {np.isnan(auxvar_vector).sum()}, Inf count: {np.isinf(auxvar_vector).sum()}")
        
        # 🔍 DEEP INVESTIGATION: Log solver settings
        v_floor = getattr(self.env, '_velocity_floor', 0.1)
        sym_eps = getattr(self.env, '_symbolic_epsilon', 1e-6)
        self.logger.info(f"🔍 Velocity floor: {v_floor} m/s, Symbolic epsilon: {sym_eps}")

        # 🔍 DEEP INVESTIGATION: Stage-wise term breakdown around observed blow-up
        if getattr(self.config.io, "log_rollout_dynamics", True):
            self._log_rollout_term_breakdown(auxvar_vector, start_stage=10, end_stage=14)
        if getattr(self.config.io, "save_rollout_csv", True):
            self._export_rollout_debug_csv(auxvar_vector, max_stage=16)
            self._export_rollout_debug_full_csv(auxvar_vector, max_stage=16)
        
        # 🔍 DIAGNOSTIC: Optional debug switches (enable when investigating NaN)
        self.coc._debug_init_guess = False
        self.coc._debug_nlp_eval = False

        #auxvar_vector = np.vstack([weights_CL, weights_CD, weights_CM])
        self.last_traj_COC = self.coc.ocSolver(horizon=111, init_state=self.init_state, auxvar_value=auxvar_vector, timeVarying=True, warm_start=True)

        if not self.last_traj_COC["success"]:
            self.logger.critical(f"⚠️ IPOPT couldn't find a solution")
            
        num_iterations = self.config.run.max_outer_iters
        iteration = downstream_info["iteration"]
        if getattr(self.config.io, "save_rollouts", True) and (iteration == 0 or iteration == (num_iterations - 1)):
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