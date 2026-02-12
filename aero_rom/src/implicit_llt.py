"""
implicit_llt.py

Implicit-differentiation (IFT/adjoint) LLT solver on top of cuNeuralFoil.

Forward:
  - Converge Gamma* with Picard in torch.no_grad().

Backward:
  - Use implicit differentiation:
      (dF/dGamma)^T lambda = dL/dGamma
    then:
      dL/dp = dL/dp_direct - lambda^T dF/dp

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import cuneuralfoil
from cuneuralfoil.main import get_aero_from_kulfan_parameters_cuda


# --- MPS dtype fix if needed ---
if hasattr(cuneuralfoil.main, "_ln_eps"):
    cuneuralfoil.main._ln_eps = cuneuralfoil.main._ln_eps.to(dtype=torch.float32)


_MODEL_SIZE_TO_ID = {
    "xxsmall": 0,
    "xsmall": 1,
    "small": 2,
    "medium": 3,
    "large": 4,
    "xlarge": 5,
    "xxlarge": 6,
    "xxxlarge": 7,
}
_ID_TO_MODEL_SIZE = {v: k for k, v in _MODEL_SIZE_TO_ID.items()}

_DEVICE_TO_ID = {"cpu": 0, "cuda": 1, "mps": 2}
_ID_TO_DEVICE = {v: k for k, v in _DEVICE_TO_ID.items()}


def _resolve_torch_device(device: str | None) -> str:
    if device is None:
        return "cpu"
    d = device.lower()
    if d in ("cuda", "cpu", "mps"):
        return d
    if d in ("metal",):
        return "mps"
    raise ValueError(f"Unknown device '{device}'. Use 'cuda', 'cpu', or 'mps'.")


@dataclass
class LLTConst:
    # geometry / wing
    dy: torch.Tensor
    y: torch.Tensor
    c: torch.Tensor
    tw: torch.Tensor
    S: torch.Tensor
    cbar: torch.Tensor
    x_c4: torch.Tensor
    span: torch.Tensor
    D_nf: torch.Tensor
    D_tr: torch.Tensor
    mirror_of: torch.Tensor

    # flow
    rho: torch.Tensor
    mu: torch.Tensor

    # solver
    n_iter: int
    beta: float
    tol: float
    enforce_symmetry: bool

    # cuNF
    model_size: str


def _eval_nf_batched(
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    alpha_eff: torch.Tensor,  # (B, n_pan)
    Re: torch.Tensor,         # (B, n_pan)
    const: LLTConst,
) -> Dict[str, torch.Tensor]:
    """
    Single cuNeuralFoil call for (B, n_pan) by flattening to (B*n_pan,).
    Keeps gradient paths to upper/lower/LE/TE.
    """
    dev = alpha_eff.device
    dev_str = dev.type

    B, n_pan = alpha_eff.shape
    alpha_flat = alpha_eff.reshape(-1)
    Re_flat = Re.reshape(-1)
    BN = alpha_flat.numel()

    # expand (no data copy; gradients accumulate correctly)
    # Expand parameters to (BN, ...) for a single cuNeuralFoil call.
    # Supports either:
    #   - global airfoil: upper/lower (8,), LE/TE scalar or (1,)
    #   - per-panel airfoil: upper/lower (n_pan, 8), LE/TE (n_pan,) or scalar
    if upper.ndim == 1:
        # global airfoil -> broadcast to all panels
        upper_batch = upper.unsqueeze(0).expand(BN, -1)
        lower_batch = lower.unsqueeze(0).expand(BN, -1)

        LE0 = LE.reshape(-1)[0]
        TE0 = TE.reshape(-1)[0]
        LE_batch = LE0.expand(BN)
        TE_batch = TE0.expand(BN)

    elif upper.ndim == 2:
        if upper.shape[0] != n_pan or lower.shape[0] != n_pan:
            raise ValueError(
                f"Per-panel Kulfan must have shape (n_pan, 8); got upper {tuple(upper.shape)}, "
                f"lower {tuple(lower.shape)} with n_pan={n_pan}"
            )

        # (n_pan, 8) -> (B, n_pan, 8) -> (BN, 8)
        upper_batch = upper.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)
        lower_batch = lower.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)

        def _to_pan(v):
            v = v.reshape(-1)
            if v.numel() == 1:
                return v[0].expand(n_pan)
            if v.numel() == n_pan:
                return v
            raise ValueError(
                f"Per-panel LE/TE must be scalar or (n_pan,), got {tuple(v.shape)} with n_pan={n_pan}"
            )

        LE_pan = _to_pan(LE)
        TE_pan = _to_pan(TE)

        LE_batch = LE_pan.unsqueeze(0).expand(B, -1).reshape(BN)
        TE_batch = TE_pan.unsqueeze(0).expand(B, -1).reshape(BN)

    else:
        raise ValueError(f"Unsupported upper.ndim={upper.ndim}; expected 1 (global) or 2 (per-panel)")

    kulfan_batch = {
        "upper_weights_cuda": upper_batch,
        "lower_weights_cuda": lower_batch,
        "leading_edge_weight_cuda": LE_batch,
        "TE_thickness_cuda": TE_batch,
    }

    aero = get_aero_from_kulfan_parameters_cuda(
        kulfan_batch,
        alpha_flat, 
        Re_flat,
        device=dev_str,
        model_size=const.model_size,
    )

    # Hard assertion: if this fails, implicit gradients cannot work
    # if not aero["CL"].requires_grad:
    #     raise RuntimeError(
    #         "cuNeuralFoil output does not require grad. "
    #         "Implicit differentiation needs cuNeuralFoil to be differentiable on this backend."
    #     )

    return {
        "CL": aero["CL"].reshape(B, n_pan),
        "CD": aero["CD"].reshape(B, n_pan),
        "CM": aero["CM"].reshape(B, n_pan),
    }


def _G(
    Gamma: torch.Tensor,     # (B, n_pan)
    alpha: torch.Tensor,     # (B, 1)
    V: torch.Tensor,         # (B, 1)
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    const: LLTConst,
) -> torch.Tensor:
    """
    One Picard update step Gamma -> Gamma_new (differentiable).
    """
    tw = const.tw.unsqueeze(0)  # (1,n_pan)
    c = const.c.unsqueeze(0)

    w_nf = Gamma @ const.D_nf.T  # (B,n_pan)

    alpha_geo = alpha + tw
    alpha_eff = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V))

    Re = const.rho * V * c / const.mu

    aero = _eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, const)
    cl = aero["CL"]

    Gamma_star = 0.5 * V * c * cl
    Gamma_new = (1.0 - const.beta) * Gamma + const.beta * Gamma_star

    if const.enforce_symmetry:
        j = const.mirror_of
        Gamma_new = 0.5 * (Gamma_new + Gamma_new[:, j])

    return Gamma_new


def _F(
    Gamma: torch.Tensor,
    alpha: torch.Tensor,
    V: torch.Tensor,
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    const: LLTConst,
) -> torch.Tensor:
    """
    Residual: F(Gamma)=Gamma - G(Gamma). Root is Gamma*.
    """
    return Gamma - _G(Gamma, alpha, V, upper, lower, LE, TE, const)


def _compute_coeffs(
    Gamma: torch.Tensor,     # (B,n_pan)
    alpha: torch.Tensor,     # (B,1)
    V: torch.Tensor,         # (B,1)
    upper: torch.Tensor,
    lower: torch.Tensor,
    LE: torch.Tensor,
    TE: torch.Tensor,
    const: LLTConst,
) -> torch.Tensor:
    """
    Returns (B,3): [CL, CD, CM_pitch]
    """
    c = const.c.unsqueeze(0)
    dy = const.dy.unsqueeze(0)
    tw = const.tw.unsqueeze(0)
    y = const.y.unsqueeze(0)
    x_c4 = const.x_c4.unsqueeze(0)

    # Trefftz induced velocity
    w_tr = Gamma @ const.D_tr.T
    w_nf= Gamma @ const.D_nf.T 

    alpha_geo = alpha + tw
    alpha_eff = alpha_geo - torch.rad2deg(torch.atan2(w_nf, V))
    Re = const.rho * V * c / const.mu

    aero = _eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re, const)
    cl = aero["CL"]
    cd = aero["CD"]
    cm = aero["CM"]

    q = 0.5 * const.rho * (V ** 2)  # (B,1)

    Lp = q * c * cl
    Dp = q * c * cd
    Di = const.rho * Gamma * w_tr #Treffz plane here for better momentum conservation - consistent with flow5 methodology

    # Integrals
    L = const.rho * V.squeeze(-1) * torch.sum(Gamma * dy, dim=-1)  # (B,)
    D = torch.sum((Dp + Di) * dy, dim=-1)

    denom = (q.squeeze(-1) * const.S).clamp_min(1e-30)
    CL = L / denom
    CD = D / denom

    # Pitching moment about the quarter-chord line (NeuralFoil CM is about c/4)
    Mprime_c4 = q * (c ** 2) * cm
    M_pitch = torch.sum(Mprime_c4 * dy, dim=-1)

    denom_pitch = (q.squeeze(-1) * const.S * const.cbar).clamp_min(1e-30)
    CM = M_pitch / denom_pitch

    return torch.stack([CL, CD, CM], dim=-1)  # (B,3)


class LLTImplicitFn(torch.autograd.Function):
    """
    Custom autograd Function:
      - forward converges Gamma* (no graph)
      - backward uses explicit Jacobian of F wrt Gamma per batch item
    """

    @staticmethod
    def forward(
        ctx,
        alpha: torch.Tensor,
        V: torch.Tensor,
        upper: torch.Tensor,
        lower: torch.Tensor,
        LE: torch.Tensor,
        TE: torch.Tensor,
        dy: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        tw: torch.Tensor,
        S: torch.Tensor,
        cbar: torch.Tensor,
        x_c4: torch.Tensor,
        span: torch.Tensor,
        D_nf: torch.Tensor,
        D_tr: torch.Tensor,
        mirror_of: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor,
        beta_t: torch.Tensor,
        tol_t: torch.Tensor,
        n_iter_t: torch.Tensor,
        enforce_sym_t: torch.Tensor,
        model_size_id: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:

        model_size = _ID_TO_MODEL_SIZE[int(model_size_id.item())]
        ctx.model_size_id = int(model_size_id.item())
        ctx.device_id = int(device_id.item())

        const = LLTConst(
            dy=dy, y=y, c=c, tw=tw, S=S, cbar=cbar, x_c4=x_c4, span=span,
            D_nf=D_nf, D_tr=D_tr, mirror_of=mirror_of,
            rho=rho, mu=mu,
            n_iter=int(n_iter_t.item()),
            beta=float(beta_t.item()),
            tol=float(tol_t.item()),
            enforce_symmetry=bool(enforce_sym_t.item() > 0.5),
            model_size=model_size,
        )

        alpha2 = alpha.reshape(-1, 1)
        V2 = V.reshape(-1, 1)
        B = alpha2.shape[0]

        with torch.no_grad():
            # Initial guess via NF at alpha_geo
            tw0 = const.tw.unsqueeze(0)
            c0 = const.c.unsqueeze(0)
            alpha_geo = alpha2 + tw0
            Re = const.rho * V2 * c0 / const.mu

            aero0 = _eval_nf_batched(upper, lower, LE, TE, alpha_geo, Re, const)
            Gamma = 0.5 * V2 * c0 * aero0["CL"]  # (B,n_pan)

            for _ in range(const.n_iter):
                Gamma_new = _G(Gamma, alpha2, V2, upper, lower, LE, TE, const)
                diff = torch.max(torch.abs(Gamma_new - Gamma))
                denom = torch.max(torch.tensor(1.0, device=Gamma.device), torch.max(torch.abs(Gamma)))
                if (diff / denom).item() < const.tol:
                    Gamma = Gamma_new
                    break
                Gamma = Gamma_new

            Gamma_star = Gamma
            C = _compute_coeffs(Gamma_star, alpha2, V2, upper, lower, LE, TE, const)

        ctx.save_for_backward(
            Gamma_star, alpha2, V2,
            upper, lower, LE, TE,
            dy, y, c, tw, S, cbar, x_c4, span, D_nf, D_tr, mirror_of, rho, mu,
            beta_t, tol_t, n_iter_t, enforce_sym_t
        )
        return C  # (B,3)

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor):
        """
        Matrix-free implicit backward using GMRES on (dF/dGamma)^T lambda = dL/dGamma.

        Removes the dense Jacobian materialization:
            J = autograd.functional.jacobian(...)
        which costs ~O(n_pan) residual evaluations and is spiky/slow.

        Instead:
        - Define JT_mv(v) = J^T v via a VJP:
              JT_mv(v) = d/dGamma <F(Gamma), v>
        - Solve for lambda with GMRES using only JT_mv.
        """
        saved = ctx.saved_tensors
        (
            Gamma_star, alpha2, V2,
            upper, lower, LE, TE,
            dy, y, c, tw, S, cbar, x_c4, span, D_nf, D_tr, mirror_of, rho, mu,
            beta_t, tol_t, n_iter_t, enforce_sym_t
        ) = saved

        model_size = _ID_TO_MODEL_SIZE[int(ctx.model_size_id)]
        const = LLTConst(
            dy=dy, y=y, c=c, tw=tw, S=S, cbar=cbar, x_c4=x_c4, span=span,
            D_nf=D_nf, D_tr=D_tr, mirror_of=mirror_of,
            rho=rho, mu=mu,
            n_iter=int(n_iter_t.item()),
            beta=float(beta_t.item()),
            tol=float(tol_t.item()),
            enforce_symmetry=bool(enforce_sym_t.item() > 0.5),
            model_size=model_size,
        )

        # -------------------------
        # Helper: GMRES (Givens)
        # -------------------------
        def gmres_solve(matvec, b, max_iter=40, tol=1e-6):
            """
            Solve A x = b using GMRES with Givens rotations.

            - Works on CPU/CUDA/MPS (no torch.linalg.solve)
            - matvec: callable(v) -> A v
            - b: (n,) tensor
            """
            n = b.numel()
            device = b.device
            dtype = b.dtype

            # x0 = 0
            x = torch.zeros_like(b)
            r0 = b - matvec(x)
            beta = torch.sqrt(torch.sum(r0 * r0))

            # If already converged
            if float(beta) < tol:
                return x

            # Krylov basis V and Hessenberg H
            V = []
            V.append(r0 / beta)

            H = torch.zeros((max_iter + 1, max_iter), device=device, dtype=dtype)
            cs = torch.zeros((max_iter,), device=device, dtype=dtype)  # cos
            sn = torch.zeros((max_iter,), device=device, dtype=dtype)  # sin

            # g is RHS in least squares problem
            g = torch.zeros((max_iter + 1,), device=device, dtype=dtype)
            g[0] = beta

            def apply_givens(h_col, k):
                # Apply previous Givens rotations to the new Hessenberg column
                for i in range(k):
                    temp = cs[i] * h_col[i] + sn[i] * h_col[i + 1]
                    h_col[i + 1] = -sn[i] * h_col[i] + cs[i] * h_col[i + 1]
                    h_col[i] = temp
                return h_col

            def make_givens(a, b):
                # Compute Givens rotation (c,s) that zeros b
                if float(b) == 0.0:
                    return torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
                r = torch.sqrt(a * a + b * b)
                c = a / r
                s = b / r
                return c, s

            k_final = -1
            for k in range(max_iter):
                # Arnoldi step
                w = matvec(V[k])

                # Modified Gram-Schmidt
                for j in range(k + 1):
                    H[j, k] = torch.sum(w * V[j])
                    w = w - H[j, k] * V[j]

                H[k + 1, k] = torch.sqrt(torch.sum(w * w))
                if float(H[k + 1, k]) != 0.0:
                    V.append(w / H[k + 1, k])
                else:
                    # happy breakdown
                    V.append(torch.zeros_like(w))

                # Apply previous Givens to this column
                h_col = H[:, k]
                h_col = apply_givens(h_col, k)

                # New Givens to zero H[k+1,k]
                c_k, s_k = make_givens(h_col[k], h_col[k + 1])
                cs[k] = c_k
                sn[k] = s_k

                # Apply to Hessenberg column
                temp = cs[k] * h_col[k] + sn[k] * h_col[k + 1]
                h_col[k + 1] = -sn[k] * h_col[k] + cs[k] * h_col[k + 1]
                h_col[k] = temp

                # Apply to g
                temp_g = cs[k] * g[k] + sn[k] * g[k + 1]
                g[k + 1] = -sn[k] * g[k] + cs[k] * g[k + 1]
                g[k] = temp_g

                # Residual norm is |g[k+1]|
                res = torch.abs(g[k + 1])
                if float(res) <= tol * float(beta):
                    k_final = k
                    break

            if k_final < 0:
                k_final = max_iter - 1

            # Solve upper triangular system R y = g (back substitution)
            m = k_final + 1
            R = H[:m, :m]
            y_ls = g[:m].clone()

            y = torch.zeros((m,), device=device, dtype=dtype)
            for i in range(m - 1, -1, -1):
                ssum = torch.sum(R[i, i + 1:m] * y[i + 1:m]) if i + 1 < m else torch.tensor(0.0, device=device, dtype=dtype)
                y[i] = (y_ls[i] - ssum) / R[i, i]

            # x = V_m @ y
            x = torch.zeros_like(b)
            for i in range(m):
                x = x + y[i] * V[i]

            return x

        # -------------------------
        # Main implicit backward
        # -------------------------
        with torch.enable_grad():
            Gamma = Gamma_star.detach().requires_grad_(True)
            grad_C = grad_C.reshape(Gamma.shape[0], 3)

            # Forward recompute coefficients (differentiable)
            C = _compute_coeffs(Gamma, alpha2, V2, upper, lower, LE, TE, const)
            L = (C * grad_C).sum()

            # RHS: dL/dGamma  (B, n_pan)
            rhs = torch.autograd.grad(L, Gamma, retain_graph=True, create_graph=False)[0]

            B, n_pan = Gamma.shape
            lambda_all = torch.zeros_like(Gamma)

            # For matrix-free VJP, we need F connected to Gamma with a graph.
            # We'll solve per batch item for simplicity (B is usually small).
            for b in range(B):
                gb = Gamma[b:b+1]  # (1, n_pan)

                def JT_mv(v: torch.Tensor) -> torch.Tensor:
                    """
                    Compute (dF/dGamma)^T v for this batch item using a VJP.
                    v: (n_pan,)
                    returns: (n_pan,)
                    """
                    v = v.reshape(1, n_pan)

                    F_b = _F(
                        gb,
                        alpha2[b:b+1],
                        V2[b:b+1],
                        upper, lower, LE, TE,
                        const,
                    )  # (1, n_pan)

                    # VJP: d/dGamma <F, v>
                    JT_v = torch.autograd.grad(
                        outputs=F_b,
                        inputs=gb,
                        grad_outputs=v,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=False,
                    )[0]  # (1, n_pan)

                    return JT_v.reshape(n_pan)

                bvec = rhs[b].detach()
                lam_b = gmres_solve(JT_mv, bvec, max_iter=min(40, n_pan), tol=1e-6)
                lambda_all[b] = lam_b

            # We do NOT want gradients flowing through GMRES internals
            lambda_all = lambda_all.detach()

            # Direct term: dL/dp (Kulfan only)
            grads_direct = torch.autograd.grad(
                L, (upper, lower, LE, TE),
                retain_graph=True, allow_unused=True, create_graph=False
            )

            # Implicit term: lambda^T dF/dp
            Fval = _F(Gamma, alpha2, V2, upper, lower, LE, TE, const)

            if not Fval.requires_grad:
                raise RuntimeError(
                    "Fval does not require grad in implicit backward. "
                    "This means F is not connected to Kulfan params in autograd."
                )

            grads_impl = torch.autograd.grad(
                Fval, (upper, lower, LE, TE),
                grad_outputs=lambda_all,
                retain_graph=False, allow_unused=True, create_graph=False
            )

            # Combine: dL/dp = direct - implicit
            out_grads = []
            for gd, gi in zip(grads_direct, grads_impl):
                if gd is None and gi is None:
                    out_grads.append(None)
                elif gd is None:
                    out_grads.append(-gi)
                elif gi is None:
                    out_grads.append(gd)
                else:
                    out_grads.append(gd - gi)

            g_upper, g_lower, g_LE, g_TE = out_grads

        # Return grads aligned with forward() inputs of LLTImplicitFn.apply(...)
        return (
            None,  # alpha
            None,  # V
            g_upper,
            g_lower,
            g_LE,
            g_TE,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None
        )



class CuNFWeissingerLLTImplicit(torch.nn.Module):
    """
    Torch module wrapper.

    forward(alpha_deg, V_inf) -> dict {"CL","CD","CM"} each shape (B,)
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

        # IMPORTANT: keep exact leaf tensors from cuKulfanAirfoil (no .to copies)
        self.upper = airfoil_cu.upper_weights_cuda
        self.lower = airfoil_cu.lower_weights_cuda
        self.LE = airfoil_cu.leading_edge_weight_cuda
        self.TE = airfoil_cu.TE_thickness_cuda

        if self.upper.device.type != self.device:
            raise RuntimeError(
                f"cuKulfanAirfoil tensors are on {self.upper.device}, but LLTImplicit device is '{self.device}'. "
                "Construct cuKulfanAirfoil with the same device."
            )

        # Const tensors
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

        self.rho = torch.as_tensor(airflow["rho"], dtype=torch.float32, device=self.device)
        self.mu = torch.as_tensor(airflow["mu"], dtype=torch.float32, device=self.device)

        self.beta_t = torch.tensor(float(beta), dtype=torch.float32, device=self.device)
        self.tol_t = torch.tensor(float(tol), dtype=torch.float32, device=self.device)
        self.n_iter_t = torch.tensor(int(n_iter), dtype=torch.float32, device=self.device)
        self.enforce_sym_t = torch.tensor(1.0 if enforce_symmetry else 0.0, dtype=torch.float32, device=self.device)

        self.model_size_id = torch.tensor(_MODEL_SIZE_TO_ID[self.model_size], dtype=torch.int64, device=self.device)
        self.device_id = torch.tensor(_DEVICE_TO_ID[self.device], dtype=torch.int64, device=self.device)

    def forward(self, alpha_deg, V_inf) -> Dict[str, torch.Tensor]:
        alpha = torch.as_tensor(alpha_deg, dtype=torch.float32, device=self.device).reshape(-1)
        V = torch.as_tensor(V_inf, dtype=torch.float32, device=self.device).reshape(-1)

        C = LLTImplicitFn.apply(
            alpha, V,
            self.upper, self.lower, self.LE, self.TE,
            self.dy, self.y, self.c, self.tw, self.S, self.cbar, self.x_c4, self.span,
            self.D_nf, self.D_tr, self.mirror_of,
            self.rho, self.mu,
            self.beta_t, self.tol_t, self.n_iter_t, self.enforce_sym_t,
            self.model_size_id, self.device_id
        )  # (B,3)

        return {"CL": C[:, 0], "CD": C[:, 1], "CM": C[:, 2]}
