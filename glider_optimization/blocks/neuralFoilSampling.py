from ..blockBase import Block
from typing import override
from ..config import Config
from ..utils.cu_kulfan_airfoil import get_aero_from_kulfan_parameters_cuda
from typing import Dict, Any
import torch
from math import sqrt   
import logging
import numpy as np
import aerosandbox as asb

# Optional 3D LLT upgrade (uses aero_rom)
try:
    from aero_rom.src.llt import LLT_computational_params as _LLT_params
    from aero_rom.src.implicit_llt import LLTImplicitFn as _LLTImplicitFn
    from aero_rom.src.implicit_llt import _MODEL_SIZE_TO_ID as _LLT_MODEL_SIZE_TO_ID
    from aero_rom.src.implicit_llt import _DEVICE_TO_ID as _LLT_DEVICE_TO_ID
except Exception:
    _LLT_params = None
    _LLTImplicitFn = None
    _LLT_MODEL_SIZE_TO_ID = None
    _LLT_DEVICE_TO_ID = None

class NeuralFoilSampling(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.device = torch.device(config.run.device)
        nfConfig = self.config.neuralFoilSampling
        
        def chebyshev_nodes(a, b, n):
            k = np.arange(n)
            return 0.5*(a+b) + 0.5*(b-a)*np.cos((2*k+1)/(2*n)*np.pi)

        n_1d = int(sqrt(nfConfig.n_samples))
        aoa_1d = torch.tensor(chebyshev_nodes(nfConfig.AoA_min, nfConfig.AoA_max, n_1d), device=self.device,dtype=torch.float32)
        re_1d  = torch.tensor(chebyshev_nodes(nfConfig.Re_min, nfConfig.Re_max, n_1d), device=self.device,dtype=torch.float32)
        
        aoa, re = torch.meshgrid(aoa_1d, re_1d, indexing="ij")
        self.alpha_batch = aoa.reshape(-1)
        self.Re_batch = re.reshape(-1)
        
        # Validation set (random uniform)
        n_val = int(nfConfig.n_samples * 0.2) # 20% validation size
        self.alpha_val = (torch.rand(n_val, device=self.device) * (nfConfig.AoA_max - nfConfig.AoA_min)) + nfConfig.AoA_min
        self.Re_val = (torch.rand(n_val, device=self.device) * (nfConfig.Re_max - nfConfig.Re_min)) + nfConfig.Re_min
        
        self.last_airfoil = None
        
        self.lambda_conf = torch.tensor(0., device=self.device, requires_grad=False)
        self.sigma = nfConfig.sigma
        self.min_confidence = nfConfig.min_confidence
        
        self.min_avg_Cl_Cd = nfConfig.min_avg_Cl_Cd
        self.lambda_clcd = torch.tensor(0., device=self.device, requires_grad=False)
        self.use_3d_llt = bool(getattr(nfConfig, "use_3d_llt", False))

        # Elevator geometry treated as fixed for now (not optimized). In 3D mode we
        # compute its coefficients once and cache them.
        self._elevator_cached = False
        self._elevator_aero_cached = None

        if self.use_3d_llt:
            if _LLT_params is None or _LLTImplicitFn is None:
                raise ImportError(
                    "use_3d_llt=True but aero_rom is not importable. "
                    "Ensure glider_optimization/aero_rom is on PYTHONPATH and deps are installed."
                )

            ckpt_path = getattr(nfConfig, "llt_ckpt_path", "aero_rom/artifacts/models/3d_blocks.pt")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            flow = ckpt.get("flow", {})
            wing = ckpt.get("wing_geometry", {})
            elev = ckpt.get("elevator_geometry", {})

            # Build LLT geometry once (airfoil name here is irrelevant: you optimize Kulfan params anyway)
            comp = _LLT_params(
                wing["y_half"], wing["c_half"], wing["xle_half"], wing["twist_half"],
                wing.get("airfoil", "naca4412").lower().replace("_", "").replace(" ", ""), # if the dict contains "airfoil" use it, otherwise default to "naca4412"
                wing.get("dihedral", 0.0),
            )

            # Const tensors
            self._llt_dy = torch.as_tensor(comp["dy"], dtype=torch.float32, device=self.device)
            self._llt_y = torch.as_tensor(comp["y_mid"], dtype=torch.float32, device=self.device)
            self._llt_c = torch.as_tensor(comp["c_mid"], dtype=torch.float32, device=self.device)
            self._llt_tw = torch.as_tensor(comp["tw_mid"], dtype=torch.float32, device=self.device)
            self._llt_S = torch.as_tensor(comp["S"], dtype=torch.float32, device=self.device)
            self._llt_cbar = torch.as_tensor(comp["cbar"], dtype=torch.float32, device=self.device)
            self._llt_x_c4 = torch.as_tensor(comp["x_c4_mid"], dtype=torch.float32, device=self.device)
            self._llt_span = torch.as_tensor(comp["span"], dtype=torch.float32, device=self.device)
            self._llt_D_nf = torch.as_tensor(comp["D_nf"], dtype=torch.float32, device=self.device)
            self._llt_D_tr = torch.as_tensor(comp["D_tr"], dtype=torch.float32, device=self.device)
            self._llt_mirror_of = torch.as_tensor(comp["mirror_of"], dtype=torch.long, device=self.device)
            # Spanwise interpolation factor (root=0 -> tip=1) for optional root/tip airfoils
            self._llt_eta = (self._llt_y.abs() / self._llt_y.abs().max().clamp_min(1e-9))
            
            rho_air = float(flow.get("rho", 1.2041))
            if "mu" in flow:
                mu_air = float(flow["mu"])
            else:
                mu_air = rho_air * float(flow.get("nu", 1.81e-5 / 1.2041))

            self._llt_rho = torch.as_tensor(rho_air, dtype=torch.float32, device=self.device)
            self._llt_mu = torch.as_tensor(mu_air, dtype=torch.float32, device=self.device)

            # Overrides (IMPORTANT: handle None cleanly)
            beta_ov = getattr(nfConfig, "llt_beta", None)
            tol_ov = getattr(nfConfig, "llt_tol", None)
            n_iter_ov = getattr(nfConfig, "llt_n_iter", None)
            enforce_sym_ov = getattr(nfConfig, "llt_enforce_symmetry", None)

            beta = float(beta_ov) if beta_ov is not None else float(ckpt.get("beta", 0.40))
            tol = float(tol_ov) if tol_ov is not None else float(ckpt.get("tol", 1e-6))
            n_iter = int(n_iter_ov) if n_iter_ov is not None else int(ckpt.get("n_iter", 15))
            enforce_sym = bool(enforce_sym_ov) if enforce_sym_ov is not None else bool(ckpt.get("enforce_symmetry", True))

            self._llt_beta_t = torch.tensor(beta, dtype=torch.float32, device=self.device)
            self._llt_tol_t = torch.tensor(tol, dtype=torch.float32, device=self.device)
            self._llt_n_iter_t = torch.tensor(float(n_iter), dtype=torch.float32, device=self.device)
            self._llt_enforce_sym_t = torch.tensor(1.0 if enforce_sym else 0.0, dtype=torch.float32, device=self.device)

            ms_ov = getattr(nfConfig, "llt_model_size", None)
            ms = ms_ov if ms_ov is not None else self.config.neuralFoilSampling.neuralFoil_size
            self._llt_model_size_id = torch.tensor(_LLT_MODEL_SIZE_TO_ID[ms], dtype=torch.int64, device=self.device)
            self._llt_device_id = torch.tensor(_LLT_DEVICE_TO_ID[self.device.type], dtype=torch.int64, device=self.device)
            
            # --- Elevator LLT context (fixed) ---
            if elev:
                comp_e = _LLT_params(
                    elev["y_half"], elev["c_half"], elev["xle_half"], elev["twist_half"],
                    elev.get("airfoil", "naca0012"), # if the dict contains "airfoil" use it, otherwise default to "naca4412"
                    elev.get("dihedral", 0.0),
                )

                self._e_llt_dy = torch.as_tensor(comp_e["dy"], dtype=torch.float32, device=self.device)
                self._e_llt_y = torch.as_tensor(comp_e["y_mid"], dtype=torch.float32, device=self.device)
                self._e_llt_c = torch.as_tensor(comp_e["c_mid"], dtype=torch.float32, device=self.device)
                self._e_llt_tw = torch.as_tensor(comp_e["tw_mid"], dtype=torch.float32, device=self.device)
                self._e_llt_S = torch.as_tensor(comp_e["S"], dtype=torch.float32, device=self.device)
                self._e_llt_cbar = torch.as_tensor(comp_e["cbar"], dtype=torch.float32, device=self.device)
                self._e_llt_x_c4 = torch.as_tensor(comp_e["x_c4_mid"], dtype=torch.float32, device=self.device)
                self._e_llt_span = torch.as_tensor(comp_e["span"], dtype=torch.float32, device=self.device)
                self._e_llt_D_nf = torch.as_tensor(comp_e["D_nf"], dtype=torch.float32, device=self.device)
                self._e_llt_D_tr = torch.as_tensor(comp_e["D_tr"], dtype=torch.float32, device=self.device)
                self._e_llt_mirror_of = torch.as_tensor(comp_e["mirror_of"], dtype=torch.long, device=self.device)

                # Fixed elevator airfoil -> fixed Kulfan params
                # Prefer YAML plane.elevator.airfoil over checkpoint metadata
                plane_elev = getattr(getattr(self.config, "plane", None), "elevator", None)

                if plane_elev is not None and getattr(plane_elev, "airfoil", None) is not None:
                    elev_airfoil_name_raw = getattr(plane_elev, "airfoil")
                else:
                    elev_airfoil_name_raw = elev.get("airfoil", "naca0012")

                elev_airfoil_name = elev_airfoil_name_raw.lower().replace("_", "").replace(" ", "")
                elev_k = asb.Airfoil(elev_airfoil_name).to_kulfan_airfoil()
                self._e_kulfan_upper = torch.as_tensor(elev_k.upper_weights, dtype=torch.float32, device=self.device)
                self._e_kulfan_lower = torch.as_tensor(elev_k.lower_weights, dtype=torch.float32, device=self.device)
                self._e_kulfan_LE = torch.as_tensor(float(getattr(elev_k, "leading_edge_weight", 0.0)), dtype=torch.float32, device=self.device)
                self._e_kulfan_TE = torch.as_tensor(float(getattr(elev_k, "TE_thickness", 0.0)), dtype=torch.float32, device=self.device)

    #@override
    # def _eval_3d_llt(self, upper, lower, LE, TE, alpha_deg: torch.Tensor, Re_ref: torch.Tensor):
    #     """
    #     Minimal 3D wrapper: LLTImplicitFn expects (alpha, V).
    #     We map Re_ref -> V via V = Re_ref * mu / (rho * cbar).
    #     """
    #     V = Re_ref * (self._llt_mu / (self._llt_rho * self._llt_cbar))

    #     C = _LLTImplicitFn.apply(
    #         alpha_deg.reshape(-1), V.reshape(-1),
    #         upper, lower, LE.reshape(-1), TE.reshape(-1),
    #         self._llt_dy, self._llt_y, self._llt_c, self._llt_tw, self._llt_S, self._llt_cbar,
    #         self._llt_x_c4, self._llt_span,
    #         self._llt_D_nf, self._llt_D_tr, self._llt_mirror_of,
    #         self._llt_rho, self._llt_mu,
    #         self._llt_beta_t, self._llt_tol_t, self._llt_n_iter_t, self._llt_enforce_sym_t,
    #         self._llt_model_size_id, self._llt_device_id,
    #     )
    #     return {"CL": C[:, 0], "CD": C[:, 1], "CM": C[:, 2]}

    @override
    def _eval_3d_llt(
        self,
        upper,
        lower,
        LE,
        TE,
        alpha_deg: torch.Tensor,
        Re_ref: torch.Tensor,
        *,
        upper_tip=None,
        lower_tip=None,
        LE_tip=None,
        TE_tip=None,
    ):
        """
        Minimal 3D wrapper: LLTImplicitFn expects (alpha, V).
        We map Re_ref -> V via V = Re_ref * mu / (rho * cbar).
        """
        # Spanwise interpolate Kulfan parameters if tip values are provided.
        # Root-only (legacy) remains the default.
        if upper_tip is not None:
            eta = self._llt_eta  # (n_pan,)
            upper = (1.0 - eta)[:, None] * upper[None, :] + eta[:, None] * upper_tip[None, :]
            lower = (1.0 - eta)[:, None] * lower[None, :] + eta[:, None] * lower_tip[None, :]
            LE = (1.0 - eta) * LE.reshape(-1)[0] + eta * LE_tip.reshape(-1)[0]
            TE = (1.0 - eta) * TE.reshape(-1)[0] + eta * TE_tip.reshape(-1)[0]

        V = Re_ref * (self._llt_mu / (self._llt_rho * self._llt_cbar))

        C = _LLTImplicitFn.apply(
            alpha_deg.reshape(-1), V.reshape(-1),
            upper, lower, LE.reshape(-1), TE.reshape(-1),
            self._llt_dy, self._llt_y, self._llt_c, self._llt_tw, self._llt_S, self._llt_cbar,
            self._llt_x_c4, self._llt_span,
            self._llt_D_nf, self._llt_D_tr, self._llt_mirror_of,
            self._llt_rho, self._llt_mu,
            self._llt_beta_t, self._llt_tol_t, self._llt_n_iter_t, self._llt_enforce_sym_t,
            self._llt_model_size_id, self._llt_device_id,
        )
        return {"CL": C[:, 0], "CD": C[:, 1], "CM": C[:, 2]}


    def _eval_3d_llt_elevator_fixed(self, alpha_deg: torch.Tensor, Re_ref: torch.Tensor):
        """3D elevator LLT evaluation using fixed (cached) Kulfan parameters."""
        V = Re_ref * (self._llt_mu / (self._llt_rho * self._e_llt_cbar))

        C = _LLTImplicitFn.apply(
            alpha_deg.reshape(-1), V.reshape(-1),
            self._e_kulfan_upper, self._e_kulfan_lower,
            self._e_kulfan_LE.reshape(-1), self._e_kulfan_TE.reshape(-1),
            self._e_llt_dy, self._e_llt_y, self._e_llt_c, self._e_llt_tw, self._e_llt_S, self._e_llt_cbar,
            self._e_llt_x_c4, self._e_llt_span,
            self._e_llt_D_nf, self._e_llt_D_tr, self._e_llt_mirror_of,
            self._llt_rho, self._llt_mu,
            self._llt_beta_t, self._llt_tol_t, self._llt_n_iter_t, self._llt_enforce_sym_t,
            self._llt_model_size_id, self._llt_device_id,
        )
        return {"CL_e": C[:, 0], "CD_e": C[:, 1], "CM_e": C[:, 2]}

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        B = self.alpha_batch.shape[0]
        
        self._last_input = downstream_info
        
        kulfan_batch = {
            "upper_weights_cuda": downstream_info["upper_weights"].repeat(B, 1),
            "lower_weights_cuda": downstream_info["lower_weights"].repeat(B, 1),
            "leading_edge_weight_cuda": downstream_info["leading_edge_weight"].repeat(B),
            "TE_thickness_cuda": downstream_info["TE_thickness"].repeat(B),
        }
        if self.use_3d_llt:
            # 3D LLT coefficients (differentiable)
            if (
                "upper_weights_tip" in downstream_info
                and "lower_weights_tip" in downstream_info
                and "leading_edge_weight_tip" in downstream_info
                and "TE_thickness_tip" in downstream_info
            ):
                self._last_aero_coeff = self._eval_3d_llt(
                    downstream_info["upper_weights"],
                    downstream_info["lower_weights"],
                    downstream_info["leading_edge_weight"],
                    downstream_info["TE_thickness"],
                    self.alpha_batch,
                    self.Re_batch,
                    upper_tip=downstream_info["upper_weights_tip"],
                    lower_tip=downstream_info["lower_weights_tip"],
                    LE_tip=downstream_info["leading_edge_weight_tip"],
                    TE_tip=downstream_info["TE_thickness_tip"],
                )
            else:
                self._last_aero_coeff = self._eval_3d_llt(
                    downstream_info["upper_weights"],
                    downstream_info["lower_weights"],
                    downstream_info["leading_edge_weight"],
                    downstream_info["TE_thickness"],
                    self.alpha_batch,
                    self.Re_batch,
                )

            conf2d = get_aero_from_kulfan_parameters_cuda(
                kulfan_batch,
                self.alpha_batch,
                self.Re_batch,
                device=self.device,
                model_size=self.config.neuralFoilSampling.neuralFoil_size,
            ).get("analysis_confidence", torch.ones_like(self.alpha_batch))
            self._last_aero_coeff["analysis_confidence"] = conf2d

            # Fixed-elevator 3D coefficients: compute once and cache (no gradient needed).
            if hasattr(self, "_e_llt_cbar") and (not self._elevator_cached):
                with torch.no_grad():
                    self._elevator_aero_cached = self._eval_3d_llt_elevator_fixed(
                        self.alpha_batch, self.Re_batch
                    )
                    # detach so it is safe to reuse
                    self._elevator_aero_cached = {k: v.detach() for k, v in self._elevator_aero_cached.items()}
                self._elevator_cached = True

        else:
            self._last_aero_coeff = get_aero_from_kulfan_parameters_cuda(
                kulfan_batch,
                self.alpha_batch,
                self.Re_batch,
                device=self.device,
                model_size=self.config.neuralFoilSampling.neuralFoil_size,
            )
        conf = self._last_aero_coeff.get("analysis_confidence")
        try:
            conf_mean = float(conf.mean().detach().cpu().item())
        except Exception:
            conf_mean = 1.0

        constraint_violation = max(0.0, self.min_confidence - conf_mean)
        lambda_val = float(self.lambda_conf.detach().cpu().item()) if isinstance(self.lambda_conf, torch.Tensor) else float(self.lambda_conf)
        aug_lagrangian = lambda_val * constraint_violation + 0.5 * float(self.sigma) * (constraint_violation ** 2)

        # Cl/Cd constraint
        CL_fwd = self._last_aero_coeff["CL"].detach()
        CD_fwd = self._last_aero_coeff["CD"].detach()
        CD_safe_fwd = torch.clamp(CD_fwd, min=1e-5)
        cl_cd_mean = float((CL_fwd / CD_safe_fwd).mean().cpu().item())
        
        violation_clcd = max(0.0, self.min_avg_Cl_Cd - cl_cd_mean)
        lambda_clcd_val = float(self.lambda_clcd.detach().cpu().item()) if isinstance(self.lambda_clcd, torch.Tensor) else float(self.lambda_clcd)
        aug_lagrangian += lambda_clcd_val * violation_clcd + 0.5 * float(self.sigma) * (violation_clcd ** 2)

        # Validation forward pass
        B_val = self.alpha_val.shape[0]
        kulfan_batch_val = {
            "upper_weights_cuda": downstream_info["upper_weights"].repeat(B_val, 1),
            "lower_weights_cuda": downstream_info["lower_weights"].repeat(B_val, 1),
            "leading_edge_weight_cuda": downstream_info["leading_edge_weight"].repeat(B_val),
            "TE_thickness_cuda": downstream_info["TE_thickness"].repeat(B_val),
        }
        if self.use_3d_llt:
            val_aero = self._eval_3d_llt(
                downstream_info["upper_weights"],
                downstream_info["lower_weights"],
                downstream_info["leading_edge_weight"],
                downstream_info["TE_thickness"],
                self.alpha_val,
                self.Re_val,
            )
            val_conf2d = get_aero_from_kulfan_parameters_cuda(
                kulfan_batch_val,
                self.alpha_val,
                self.Re_val,
                device=self.device,
                model_size=self.config.neuralFoilSampling.neuralFoil_size,
            ).get("analysis_confidence", torch.ones_like(self.alpha_val))
            val_aero["analysis_confidence"] = val_conf2d
        else:
            val_aero = get_aero_from_kulfan_parameters_cuda(
                kulfan_batch_val,
                self.alpha_val,
                self.Re_val,
                device=self.device,
                model_size=self.config.neuralFoilSampling.neuralFoil_size,
            )

        return {
            "alpha": self.alpha_batch,
            "Re": self.Re_batch,
            "CL": self._last_aero_coeff["CL"].detach(),
            "CD": self._last_aero_coeff["CD"].detach(),
            "CM": self._last_aero_coeff["CM"].detach(),
            # Optional fixed-elevator outputs (present only when 3D is enabled and elevator geometry exists)
            **(self._elevator_aero_cached if (self.use_3d_llt and self._elevator_cached and self._elevator_aero_cached is not None) else {}),
            "augmented_lagrangian": aug_lagrangian,
            # Validation data
            "val_alpha": self.alpha_val,
            "val_Re": self.Re_val,
            "val_CL": val_aero["CL"].detach(),
            "val_CD": val_aero["CD"].detach(),
            "val_CM": val_aero["CM"].detach(),
            "iteration": downstream_info["iteration"]
        }

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        CL = self._last_aero_coeff["CL"]
        CD = self._last_aero_coeff["CD"]
        CM = self._last_aero_coeff["CM"]
        conf = self._last_aero_coeff["analysis_confidence"]
        
        constraint = self.min_confidence - conf.mean() 
        constraint_violation = torch.relu(constraint)
        constraint_lagrangian = self.lambda_conf * constraint_violation + self.sigma/2 * (constraint_violation**2)

        # Cl/Cd constraint backward
        CD_safe = torch.clamp(CD, min=1e-5)
        cl_cd_ratio = CL / CD_safe
        constraint_clcd = self.min_avg_Cl_Cd - cl_cd_ratio.mean()
        violation_clcd = torch.relu(constraint_clcd)
        constraint_lagrangian_clcd = self.lambda_clcd * violation_clcd + self.sigma/2 * (violation_clcd**2)
        
        constraint_lagrangian = constraint_lagrangian + constraint_lagrangian_clcd
        
        if violation_clcd.detach() > 1.0:
             self.logger.warning(f"⚠️ Large Cl/Cd violation. Mean: {cl_cd_ratio.mean().detach():.3f}. Target: {self.min_avg_Cl_Cd:.3f}")

        if constraint_violation.detach() > 0.1:
            self.logger.critical(f"⚠️ Large confidence violation detected. Training may become unstable. Mean Confidence {conf.mean():.3f}. Target {self.min_confidence:.3f}")
        if CL.isnan().any():
            self.logger.critical("⚠️ NaN detected in NeuralFoilSampling feedforward CL")
        if CD.isnan().any():
            self.logger.critical("⚠️ NaN detected in NeuralFoilSampling feedforward CD")
        if CM.isnan().any():
            self.logger.critical("⚠️ NaN detected in NeuralFoilSampling feedforward CM")
                
        dJ_dy = upstream_grads["dJ_dy"]
        
        upper = self._last_input["upper_weights"]
        lower = self._last_input["lower_weights"]
        LE = self._last_input["leading_edge_weight"]
        TE = self._last_input["TE_thickness"]        
                
        Y = torch.cat([CL, CD, CM], dim=0)
        
        grad_lagrangian = torch.autograd.grad(constraint_lagrangian, [upper, lower, LE, TE], retain_graph = True )
        grad = torch.autograd.grad(Y, [upper, lower, LE, TE], grad_outputs=dJ_dy.flatten())
        
        if grad[0].isnan().any():
            self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[0]")
        if grad[1].isnan().any():
            self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[1]")
        if grad[2].isnan().any():
            self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[2]")
        if grad[3].isnan().any():
            self.logger.critical(f"⚠️ NaN detected in NeuralFoilSampling backward grad[3]")
            
        with torch.no_grad():
            self.lambda_conf += self.sigma * constraint_violation.mean().detach()
            self.lambda_clcd += self.sigma * violation_clcd.mean().detach()
            
        return {
            "dupper_params": grad[0] + grad_lagrangian[0],
            "dlower_params": grad[1] + grad_lagrangian[1],
            "dleading_edge_param": grad[2] + grad_lagrangian[2],
            "dTE_thickness_param": grad[3] + grad_lagrangian[3],
        }
