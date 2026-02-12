# Differentiable lifting-line solver using cuNeuralFoil (Torch only).
#
# This mirrors the logic of run_llt_cuNF_grid in llt.py, but implemented fully
# in PyTorch so that gradients can propagate from 3D coefficients back to
# cuKulfanAirfoil shape parameters (Kulfan weights).

from __future__ import annotations

import torch
import cuneuralfoil
from cuneuralfoil.main import get_aero_from_kulfan_parameters_cuda


# --- MPS fix: ensure _ln_eps is float32, not float64 (same as in llt.py) ---
if hasattr(cuneuralfoil.main, "_ln_eps"):
    cuneuralfoil.main._ln_eps = cuneuralfoil.main._ln_eps.to(dtype=torch.float32)


def _resolve_torch_device(device: str | None) -> str:
    """
    Map user-facing device string to a torch device string.

    - "cuda" -> "cuda"
    - "cpu"  -> "cpu"
    - "mps"  -> "mps" (Apple Silicon, Metal backend)
    """
    if device is None:
        return "cpu"

    d = device.lower()
    if d == "cuda":
        return "cuda"
    if d == "cpu":
        return "cpu"
    if d in ["mps", "metal"]:
        return "mps"

    raise ValueError(f"Unknown device string '{device}'. Use 'cuda', 'cpu', or 'mps'.")


class CuNFWeissingerLLT(torch.nn.Module):
    """
    Differentiable 3D LLT model on top of cuNeuralFoil.

    Usage
    -----
    - Precompute computation_params with LLT_computational_params(...) in src.llt
      (planform geometry, influence matrices, etc. in NumPy).
    - Wrap an AeroSandbox KulfanAirfoil in cuKulfanAirfoil(requires_grad=True).
    - Build this module with those two objects and flow properties.
    - Call forward(alpha_deg, V_inf) to get 3D CL, CD, CM, etc. as torch.Tensors.
    - Use autograd (tensor.backward() or torch.autograd.grad) to get gradients
      w.r.t. Kulfan weights and/or alpha/V.

    This is meant for local gradient-based optimisation / sensitivity analysis.
    For bulk dataset generation, keep using src.llt.run_llt_cuNF_grid (NumPy).
    """

    def __init__(
        self,
        airfoil_cu,
        computation_params: dict,
        airflow: dict,
        n_iter: int = 30,
        beta: float = 0.40,
        tol: float = 1e-4,
        enforce_symmetry: bool = True,
        device: str = "cuda",
        model_size: str = "xlarge",
    ) -> None:
        super().__init__()

        self.device = _resolve_torch_device(device)
        self.model_size = model_size
        self.n_iter = int(n_iter)
        self.beta = float(beta)
        self.tol = float(tol)
        self.enforce_symmetry = bool(enforce_symmetry)

        # cuKulfanAirfoil object (Kulfan weights live here as torch tensors)
        self.airfoil_cu = airfoil_cu

        # Geometry / planform (converted once from NumPy to torch, no gradients here)
        self.dy = torch.as_tensor(computation_params["dy"], dtype=torch.float32, device=self.device)
        self.y = torch.as_tensor(computation_params["y_mid"], dtype=torch.float32, device=self.device)
        self.c = torch.as_tensor(computation_params["c_mid"], dtype=torch.float32, device=self.device)
        self.tw = torch.as_tensor(computation_params["tw_mid"], dtype=torch.float32, device=self.device)

        self.S = torch.as_tensor(computation_params["S"], dtype=torch.float32, device=self.device)
        self.cbar = torch.as_tensor(computation_params["cbar"], dtype=torch.float32, device=self.device)
        self.x_c4 = torch.as_tensor(computation_params["x_c4_mid"], dtype=torch.float32, device=self.device)
        self.span = torch.as_tensor(computation_params["span"], dtype=torch.float32, device=self.device)

        self.D_nf = torch.as_tensor(computation_params["D_nf"], dtype=torch.float32, device=self.device)
        self.D_tr = torch.as_tensor(computation_params["D_tr"], dtype=torch.float32, device=self.device)
        self.mirror_of = torch.as_tensor(computation_params["mirror_of"], dtype=torch.long, device=self.device)

        # Flow properties
        self.rho = torch.as_tensor(airflow["rho"], dtype=torch.float32, device=self.device)
        self.mu = torch.as_tensor(airflow["mu"], dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    #        cuNeuralFoil evaluation in Torch (no torch.no_grad)        #
    # ------------------------------------------------------------------ #
    def _eval_nf(self, alpha_2d: torch.Tensor, Re_2d: torch.Tensor) -> dict:
        """
        Evaluate cuNeuralFoil on a 2D grid of (alpha, Re).

        Parameters
        ----------
        alpha_2d : (B, n_pan) tensor of angles of attack in degrees.
        Re_2d    : (B, n_pan) tensor of Reynolds numbers.

        Returns
        -------
        dict with keys like "CL", "CD", "CM", each a tensor of shape (B, n_pan).
        """
        alpha_flat = alpha_2d.reshape(-1)
        Re_flat = Re_2d.reshape(-1)
        B = alpha_flat.shape[0]

        # Base Kulfan parameters (already on device, possibly requires_grad=True)
        upper_base = self.airfoil_cu.upper_weights_cuda.to(self.device)
        lower_base = self.airfoil_cu.lower_weights_cuda.to(self.device)
        LE_base = self.airfoil_cu.leading_edge_weight_cuda.to(self.device)
        TE_base = self.airfoil_cu.TE_thickness_cuda.to(self.device)

        # Geometry batch: repeat the same airfoil B times
        upper_batch = upper_base.unsqueeze(0).repeat(B, 1)
        lower_batch = lower_base.unsqueeze(0).repeat(B, 1)
        LE_batch = LE_base.repeat(B)
        TE_batch = TE_base.repeat(B)

        kulfan_batch = {
            "upper_weights_cuda": upper_batch,
            "lower_weights_cuda": lower_batch,
            "leading_edge_weight_cuda": LE_batch,
            "TE_thickness_cuda": TE_batch,
        }

        # No torch.no_grad() here: we want gradients through cuNeuralFoil and LLT
        aero_t = get_aero_from_kulfan_parameters_cuda(
            kulfan_batch,
            alpha_flat,
            Re_flat,
            device=self.device,
            model_size=self.model_size,
        )

        out = {}
        for k, v in aero_t.items():
            out[k] = v.reshape(alpha_2d.shape)
        return out

    # ------------------------------------------------------------------ #
    #                     Main forward LLT computation                   #
    # ------------------------------------------------------------------ #
    def forward(self, alpha_deg, V_inf) -> dict:
        """
        Forward pass of the 3D LLT model.

        Parameters
        ----------
        alpha_deg : float or 1D array-like
            Geometric angle(s) of attack in degrees.
        V_inf : float or 1D array-like
            Freestream speed(s) in m/s.

        Returns
        -------
        dict with torch tensors (shape (B,) if B inputs):
          - "CL", "CD", "CDp", "CDi"
          - "CM"  (pitching moment coefficient about x_ref)
          - "CMx" (roll coefficient)
          - "CMz" (yaw coefficient)
          - "L", "Dp", "Di"   (optional integrals)
          - "M_pitch", "M_roll", "M_yaw"
          - "CONF_MIN" (min confidence over span, if provided by cuNeuralFoil)
        """
        # Convert inputs to 1D tensors on the right device
        alpha_t = torch.as_tensor(alpha_deg, dtype=torch.float32, device=self.device)
        V_t = torch.as_tensor(V_inf, dtype=torch.float32, device=self.device)

        alpha_flat = alpha_t.reshape(-1)  # (B,)
        V_flat = V_t.reshape(-1)          # (B,)
        B = alpha_flat.shape[0]

        # Expand to (B, n_pan)
        aoa_2d = alpha_flat.unsqueeze(-1)  # (B, 1)
        V_2d = V_flat.unsqueeze(-1)        # (B, 1)

        tw = self.tw.unsqueeze(0)       # (1, n_pan)
        c = self.c.unsqueeze(0)         # (1, n_pan)
        dy = self.dy.unsqueeze(0)       # (1, n_pan)
        y = self.y.unsqueeze(0)         # (1, n_pan)
        x_c4 = self.x_c4.unsqueeze(0)   # (1, n_pan)

        # Geometric alpha and Reynolds per panel
        alpha_geo = aoa_2d + tw                 # (B, n_pan)
        Re_panels = self.rho * V_2d * c / self.mu

        # Initial near-field lookup and Gamma guess
        aero0 = self._eval_nf(alpha_geo, Re_panels)
        cl = aero0["CL"]                        # (B, n_pan)
        Gamma = 0.5 * V_2d * c * cl            # (B, n_pan)

        # Influence application: gamma @ D^T
        def apply_D(mat: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
            return gamma @ mat.T

        # Picard iteration
        for _ in range(self.n_iter):
            w_nf = apply_D(self.D_nf, Gamma)  # (B, n_pan)
            alpha_eff_iter = alpha_geo - torch.rad2deg(
                torch.atan2(w_nf, V_2d)
            )

            aer = self._eval_nf(alpha_eff_iter, Re_panels)
            cl_star = aer["CL"]
            Gamma_star = 0.5 * V_2d * c * cl_star

            Gamma_new = (1.0 - self.beta) * Gamma + self.beta * Gamma_star

            if self.enforce_symmetry:
                j = self.mirror_of  # (n_pan,)
                Gamma_new = 0.5 * (Gamma_new + Gamma_new[:, j])

            diff = torch.max(torch.abs(Gamma_new - Gamma))
            max_gamma = torch.max(torch.abs(Gamma))
            denom = torch.maximum(torch.tensor(1.0, device=self.device), max_gamma)
            if float(diff / denom) < self.tol:
                Gamma = Gamma_new
                break

            Gamma = Gamma_new

        # Final downwash & Trefftz induced velocity
        w_nf = apply_D(self.D_nf, Gamma)
        w_tr = apply_D(self.D_tr, Gamma)

        alpha_eff = alpha_geo - torch.rad2deg(
            torch.atan2(w_nf, V_2d)
        )

        # Final section aerodynamics
        aer_final = self._eval_nf(alpha_eff, Re_panels)
        cl = aer_final["CL"]
        cd = aer_final["CD"]
        cm = aer_final["CM"]

        # Confidence: min over span, keep percent convention if available
        conf = aer_final.get(
            "analysis_confidence_percent",
            aer_final.get("analysis_confidence", None),
        )
        if conf is None:
            CONF_MIN = torch.full((B,), 100.0, dtype=torch.float32, device=self.device)
        else:
            max_conf = torch.max(conf)
            # If max_conf <= 1, it's likely 0–1; rescale to %
            if float(max_conf) <= 1.0:
                conf_pct = 100.0 * conf
            else:
                conf_pct = conf
            CONF_MIN = torch.min(conf_pct, dim=-1).values

        # Per-unit-span loads
        q_inf = 0.5 * self.rho * (V_2d ** 2)  # (B, 1)

        Lp = q_inf * c * cl
        Dp_prime = q_inf * c * cd
        Di_prime = self.rho * Gamma * w_tr
        D_total = Dp_prime + Di_prime

        # Integrals over span
        L = self.rho * V_flat * torch.sum(Gamma * dy, dim=-1)  # (B,)
        Dp = torch.sum(Dp_prime * dy, dim=-1)
        Di = torch.sum(Di_prime * dy, dim=-1)

        # 3D coefficients
        denom_S = q_inf.squeeze(-1) * torch.maximum(self.S, torch.tensor(1e-30, device=self.device))
        CL = L / denom_S
        CDp = Dp / denom_S
        CDi = Di / denom_S
        CD = CDp + CDi

        # Pitching moment about the quarter-chord line (NeuralFoil CM is about c/4)
        Mprime_c4 = q_inf * (c ** 2) * cm
        M_pitch = torch.sum(Mprime_c4 * dy, dim=-1)
        denom_pitch = q_inf.squeeze(-1) * self.S * self.cbar
        CM_total = M_pitch / torch.maximum(denom_pitch, torch.tensor(1e-30, device=self.device))

        # Roll (x) and yaw (z) moments
        M_roll = torch.sum(y * Lp * dy, dim=-1)      # ∫ y L' dy
        M_yaw = torch.sum(y * D_total * dy, dim=-1)  # ∫ y D' dy

        denom_span = q_inf.squeeze(-1) * torch.maximum(self.S * self.span, torch.tensor(1e-30, device=self.device))
        CMx = M_roll / denom_span
        CMz = M_yaw / denom_span

        return {
            "CL": CL,
            "CD": CD,
            "CDp": CDp,
            "CDi": CDi,
            "CM": CM_total,
            "CMx": CMx,
            "CMz": CMz,
            "L": L,
            "Dp": Dp,
            "Di": Di,
            "M_pitch": M_pitch,
            "M_roll": M_roll,
            "M_yaw": M_yaw,
            "CONF_MIN": CONF_MIN,
        }
