from typing import Any, Dict, override

import numpy as np
import torch
import wandb

from ..config import Config
from ..utils import llt
from ..utils.spanwise_geometry import build_half_wing_stations_from_cfg, mix_root_tip
from .neuralFoilSampling import NeuralFoilSampling


class NeuralFoilSampling3D(NeuralFoilSampling):
    """Spanwise (lifting-line) replacement for NeuralFoilSampling.

    Instead of sampling the NeuralFoil surrogate directly at each (alpha, Re) grid
    point, this solves the LLT fixed point over the wing's panels
    """

    _N_SPAN_STATIONS = 7

    @override
    def __init__(self, config: Config):
        super().__init__(config)
        nfConfig = config.neuralFoilSampling

        wing_cfg = (getattr(config, "plane", {}) or {}).get("wing", {})
        stations = build_half_wing_stations_from_cfg(wing_cfg, n_span_stations=self._N_SPAN_STATIONS)
        comp = llt.build_llt_system(
            stations["y_half"], stations["c_half"], stations["xle_half"], stations["twist_half"],
            dihedral_deg=stations["dihedral_deg"],
        )

        def T(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=self.device)

        self.llt_dy, self.llt_c, self.llt_tw = T(comp["dy"]), T(comp["c_mid"]), T(comp["tw_mid"])
        self.llt_S, self.llt_cbar = T(comp["S"]), T(comp["cbar"])
        self.llt_D_nf, self.llt_D_tr = T(comp["D_nf"]), T(comp["D_tr"])
        self.llt_mirror = T(comp["mirror_of"], dtype=torch.long)
        self.llt_cos_sw, self.llt_cos2_sw = T(comp["cos_sweep"]), T(comp["cos2_sweep"])
        self.llt_rho, self.llt_mu = T(1.225), T(1.789e-5)

        half_span = max(comp["span"] * 0.5, 1e-9)
        self.llt_eta = T(np.clip(np.abs(comp["y_mid"]) / half_span, 0.0, 1.0))

        self.llt_beta = float(nfConfig.llt_beta)
        self.llt_tol = float(nfConfig.llt_tol)
        self.llt_max_iter = int(nfConfig.llt_max_iter)
        self.model_size = nfConfig.neuralFoil_size

    def _eval_3d_llt(self, upper, lower, LE, TE, alpha_deg, Re_ref, upper_tip, lower_tip, LE_tip, TE_tip):
        """LLTImplicitFn expects (alpha, V); V is recovered from the sampled reference
        Reynolds number via V = Re * mu / (rho * cbar).

        upper/lower/LE/TE (root and tip) are the airfoil design parameters, owned by
        Airfoil3D and kept on CPU there (matching Airfoil's convention) so gradients
        flow back to them on their own device; the device transfer for the actual LLT
        compute happens here, right before it's needed, the same way
        get_aero_from_kulfan_parameters_cuda does it for the 2D pipeline.
        """
        alpha_deg = alpha_deg.to(torch.float32)
        Re_ref = Re_ref.to(torch.float32)
        upper, lower = upper.to(self.device), lower.to(self.device)
        LE, TE = LE.to(self.device), TE.to(self.device)
        upper_tip, lower_tip = upper_tip.to(self.device), lower_tip.to(self.device)
        LE_tip, TE_tip = LE_tip.to(self.device), TE_tip.to(self.device)

        eta = self.llt_eta
        upper = mix_root_tip(upper, upper_tip, eta)
        lower = mix_root_tip(lower, lower_tip, eta)
        LE = mix_root_tip(LE, LE_tip, eta)
        TE = mix_root_tip(TE, TE_tip, eta)
        V = Re_ref * (self.llt_mu / (self.llt_rho * self.llt_cbar))

        C, alpha_eff_pan, Re_pan = llt.LLTImplicitFn.apply(
            alpha_deg, V, upper, lower, LE, TE,
            self.llt_dy, self.llt_c, self.llt_tw, self.llt_S, self.llt_cbar,
            self.llt_D_nf, self.llt_D_tr, self.llt_mirror, self.llt_cos_sw, self.llt_cos2_sw,
            self.llt_rho, self.llt_mu,
            self.llt_beta, self.llt_tol, self.llt_max_iter, True, self.model_size,
        )

        aero_conf = llt.eval_nf_batched(upper, lower, LE, TE, alpha_eff_pan, Re_pan, self.model_size)
        conf_mean = aero_conf["analysis_confidence"].mean(dim=1)

        return {"CL": C[:, 0], "CD": C[:, 1], "CM": C[:, 2], "analysis_confidence": conf_mean}

    def _eval_from(self, downstream_info, alpha, Re):
        return self._eval_3d_llt(
            downstream_info["upper_weights"], downstream_info["lower_weights"],
            downstream_info["leading_edge_weight"], downstream_info["TE_thickness"],
            alpha, Re,
            downstream_info["upper_weights_tip"], downstream_info["lower_weights_tip"],
            downstream_info["leading_edge_weight_tip"], downstream_info["TE_thickness_tip"],
        )

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self._last_input = downstream_info
        self._last_aero_coeff = self._eval_from(downstream_info, self.alpha_batch, self.Re_batch)

        conf_mean = float(self._last_aero_coeff["analysis_confidence"].mean().item())
        constraint_violation = max(0.0, self.min_confidence - conf_mean)
        lambda_val = float(self.lambda_conf.item())
        aug_lagrangian = lambda_val * constraint_violation + 0.5 * self.rho * constraint_violation ** 2

        CL_fwd = self._last_aero_coeff["CL"].detach()
        CD_fwd = self._last_aero_coeff["CD"].detach()
        cl_cd_mean = float((CL_fwd / CD_fwd.clamp(min=1e-5)).mean().item())
        violation_clcd = max(0.0, self.min_avg_Cl_Cd - cl_cd_mean)
        lambda_clcd_val = float(self.lambda_clcd.item())
        aug_lagrangian += lambda_clcd_val * violation_clcd + 0.5 * self.rho * violation_clcd ** 2

        if self.config.io.wandb.enabled:
            wandb.log(
                {"lagrangian/lambda_conf": lambda_val, "lagrangian/lambda_clcd": lambda_clcd_val},
                step=downstream_info["iteration"],
            )

        with torch.no_grad():
            val_aero = self._eval_from(downstream_info, self.alpha_val, self.Re_val)

        out = {
            "alpha": self.alpha_batch,
            "Re": self.Re_batch,
            "CL": self._last_aero_coeff["CL"].detach(),
            "CD": self._last_aero_coeff["CD"].detach(),
            "CM": self._last_aero_coeff["CM"].detach(),
            "augmented_lagrangian": aug_lagrangian,
            "val_alpha": self.alpha_val,
            "val_Re": self.Re_val,
            "val_CL": val_aero["CL"].detach(),
            "val_CD": val_aero["CD"].detach(),
            "val_CM": val_aero["CM"].detach(),
            "iteration": downstream_info["iteration"],
            "wing_geometry": {"S_w": float(self.llt_S.item()), "chord": float(self.llt_cbar.item())},
        }
        if "wing_centroid_offset" in downstream_info:
            out["wing_geometry"]["centroid_offset"] = downstream_info["wing_centroid_offset"]
        return out

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        CL = self._last_aero_coeff["CL"]
        CD = self._last_aero_coeff["CD"]
        CM = self._last_aero_coeff["CM"]
        conf = self._last_aero_coeff["analysis_confidence"]

        constraint_violation = torch.relu(self.min_confidence - conf.mean())
        constraint_lagrangian = self.lambda_conf * constraint_violation + self.rho / 2 * constraint_violation ** 2

        cl_cd_ratio = CL / CD.clamp(min=1e-5)
        violation_clcd = torch.relu(self.min_avg_Cl_Cd - cl_cd_ratio.mean())
        constraint_lagrangian = constraint_lagrangian + (
            self.lambda_clcd * violation_clcd + self.rho / 2 * violation_clcd ** 2
        )

        if violation_clcd.detach() > 1.0:
            self.logger.warning(
                "Large Cl/Cd violation: mean %.3f, target %.3f", cl_cd_ratio.mean().item(), self.min_avg_Cl_Cd
            )
        if constraint_violation.detach() > 0.1:
            self.logger.critical(
                "Large confidence violation: mean %.3f, target %.3f", conf.mean().item(), self.min_confidence
            )

        params = [
            self._last_input["upper_weights"], self._last_input["lower_weights"],
            self._last_input["leading_edge_weight"], self._last_input["TE_thickness"],
            self._last_input["upper_weights_tip"], self._last_input["lower_weights_tip"],
            self._last_input["leading_edge_weight_tip"], self._last_input["TE_thickness_tip"],
        ]
        Y = torch.cat([CL, CD, CM], dim=0)
        dJ_dy = upstream_grads["dJ_dy"]

        grad_lagrangian = torch.autograd.grad(constraint_lagrangian, params, retain_graph=True)
        grad = torch.autograd.grad(Y, params, grad_outputs=dJ_dy.flatten())

        for i, g in enumerate(grad):
            if g.isnan().any():
                self.logger.critical("NaN detected in NeuralFoilSampling3D backward grad[%d]", i)

        with torch.no_grad():
            self.lambda_conf += self.rho * constraint_violation.detach()
            self.lambda_clcd += self.rho * violation_clcd.detach()

        keys = [
            "dupper_params", "dlower_params", "dleading_edge_param", "dTE_thickness_param",
            "dupper_params_tip", "dlower_params_tip", "dleading_edge_param_tip", "dTE_thickness_param_tip",
        ]
        return {k: g + gl for k, g, gl in zip(keys, grad, grad_lagrangian)}
