from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, List
import torch
import torch.nn as nn
import aerosandbox as asb
from glider_optimization.utils.cu_kulfan_airfoil import get_aero_from_kulfan_parameters_cuda
from glider_optimization.config import load_config
from glider_optimization.blocks import NeuralFoilSampling, ReducedModel, OCP, Evaluation
from glider_optimization.logger import setup_logging
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from math import sqrt

# a QuantityFn takes (aero_dict, alpha_batch, Re_batch) and returns a scalar
QuantityFn = Callable[[dict, torch.Tensor, torch.Tensor], torch.Tensor]

@dataclass
class Objective:
    name: str
    fn: QuantityFn


@dataclass
class Constraint:
    name: str
    fn: QuantityFn
    kind: str  # 'ge' | 'le' | 'eq'
    target: float
    mu: float = 10.0
    lam: float = 0.0 

    def violation(self, quantity: torch.Tensor) -> torch.Tensor:
        if self.kind == "ge":
            return torch.relu(self.target - quantity)
        elif self.kind == "le":
            return torch.relu(quantity - self.target)
        elif self.kind == "eq":
            return quantity - self.target
        else:
            raise ValueError(f"Unknown constraint kind '{self.kind}' for '{self.name}'")

    def lagrangian_term(self, quantity: torch.Tensor) -> torch.Tensor:
        v = self.violation(quantity)
        if self.kind == "eq":
            # standard equality augmented Lagrangian: linear + quadratic, no relu
            return self.lam * v + 0.5 * self.mu * v ** 2
        else:
            return self.lam * v + 0.5 * self.mu * v ** 2

    def update_multiplier(self, quantity: torch.Tensor):
        with torch.no_grad():
            v = self.violation(quantity)
            if self.kind == "ge" or self.kind == "le":
                # keep lambda >= 0 for inequality constraints
                self.lam = max(0.0, self.lam + self.mu * v.mean().item())
            else:
                self.lam = self.lam + self.mu * v.mean().item()


@dataclass
class DesignVariables:
    upper: nn.Parameter
    lower: nn.Parameter
    LE: nn.Parameter
    TE: nn.Parameter

    def as_list(self):
        return [self.upper, self.lower, self.LE, self.TE]

class AirfoilDesignProblem:
    def __init__(
        self,
        device: str,
        objective: Objective,
        constraints: List[Constraint],
        model_size: str = "xxxlarge",
        batch_side: int = 8,
        aoa_range=(-20.0, 30.0),
        re_range=(100.0, 100000.0),
        lr: float = 1e-4,
        min_thickness_gap: float = 0.002,
        seed_airfoil: str = "naca4412",
    ):
        self.device = device
        self.objective = objective
        self.constraints = constraints
        self.model_size = model_size
        self.min_thickness_gap = min_thickness_gap

        airfoil_asb = asb.Airfoil(seed_airfoil).to_kulfan_airfoil()
        self.vars = DesignVariables(
            upper=nn.Parameter(torch.tensor(airfoil_asb.upper_weights, device=device)),
            lower=nn.Parameter(torch.tensor(airfoil_asb.lower_weights, device=device)),
            LE=nn.Parameter(torch.tensor(airfoil_asb.leading_edge_weight, device=device)),
            TE=nn.Parameter(torch.tensor(airfoil_asb.TE_thickness, device=device)),
        )
        self.optimizer = torch.optim.Adam(self.vars.as_list(), lr=lr)

        self.B = batch_side ** 2
        aoa_1d = torch.linspace(*aoa_range, batch_side, device=device)
        re_1d = torch.linspace(*re_range, batch_side, device=device)
        aoa, re = torch.meshgrid(aoa_1d, re_1d, indexing="ij")
        self.alpha_batch = aoa.reshape(-1)
        self.re_batch = re.reshape(-1)

    def _query_aero(self) -> dict:
        v = self.vars
        B = self.B
        kulfan_batch = {
            "upper_weights_cuda": v.upper.unsqueeze(0).repeat(B, 1),
            "lower_weights_cuda": v.lower.unsqueeze(0).repeat(B, 1),
            "leading_edge_weight_cuda": v.LE.unsqueeze(0).repeat(B),
            "TE_thickness_cuda": v.TE.unsqueeze(0).repeat(B),
        }
        return get_aero_from_kulfan_parameters_cuda(
            kulfan_batch, self.alpha_batch, self.re_batch,
            device=self.device, model_size=self.model_size,
        )

    def step(self):
        aero = self._query_aero()

        loss = self.objective.fn(aero, self.alpha_batch, self.re_batch)

        al = loss
        quantities = {}
        for c in self.constraints:
            q = c.fn(aero, self.alpha_batch, self.re_batch)
            quantities[c.name] = q
            al = al + c.lagrangian_term(q)

        grads = torch.autograd.grad(al, self.vars.as_list())
        self.optimizer.zero_grad()
        for p, g in zip(self.vars.as_list(), grads):
            p.grad = g
        self.optimizer.step()

        with torch.no_grad():
            self.vars.TE.clamp_(9, 0.01)
            self.vars.upper.copy_(
                torch.maximum(self.vars.upper, self.vars.lower + self.min_thickness_gap)
            )
            for c in self.constraints:
                c.update_multiplier(quantities[c.name])

        return {"loss": loss.detach(), "al": al.detach(), "quantities": quantities, "aero": aero}

    def train(self, n_iters: int, log_every: int = 100):
        for it in range(n_iters):
            info = self.step()
            if (it + 1) % log_every == 0:
                parts = [f"loss = {info['loss'].mean().item():.3f}", f"al = {info['al'].mean().item():.3f}"]
                for c in self.constraints:
                    parts.append(f"{c.name} = {info['quantities'][c.name].mean().item():.3f}")
                print(f"it {it + 1:05d} || " + " || ".join(parts))

    def to_kulfan_airfoil(self, name="optimized") -> asb.KulfanAirfoil:
        v = self.vars
        return asb.KulfanAirfoil(
            name=name,
            lower_weights=v.lower.detach().cpu().numpy(),
            upper_weights=v.upper.detach().cpu().numpy(),
            leading_edge_weight=v.LE.detach().cpu().numpy(),
            TE_thickness=v.TE.detach().cpu().numpy(),
        )


def masked_mean(field: torch.Tensor, alpha_batch: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    mask = (alpha_batch >= lo) & (alpha_batch <= hi)
    return field[mask].mean()


def solve_ocp_once(problem: AirfoilDesignProblem, config_path: Path, run_name: str = "single_step_eval") -> dict:
    cfg = load_config(config_path)
    cfg.run.device = problem.device
    cfg.io.run_name = run_name
    cfg.io.checkpoint_dir = f"./runs/{run_name}"
    cfg.io.wandb.enabled = False
    setup_logging(cfg.io)

    blocks = [NeuralFoilSampling(cfg), ReducedModel(cfg), OCP(cfg), Evaluation(cfg)]

    v = problem.vars
    data = {
        "upper_weights": v.upper,
        "lower_weights": v.lower,
        "leading_edge_weight": v.LE,
        "TE_thickness": v.TE,
        "iteration": 0,
    }
    for block in blocks:
        data = block.forward(data)

    return data

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # AoA band the perching flare actually operates in; unlike a plain full-range
    # mean, this keeps the reward tied to the regime the OCP needs to be good at.
    CORE_AOA_LO, CORE_AOA_HI = -6.0, 16.0

    objective = Objective(
        name="neg_drag_in_core_range",
        fn=lambda aero, alpha, re: -masked_mean(aero["CL"] / aero["CD"], alpha, CORE_AOA_LO, CORE_AOA_HI),
    )

    constraints = [
        Constraint(
            name="confidence",
            fn=lambda aero, alpha, re: aero["analysis_confidence"].mean(),
            kind="ge",
            target=0.7,
            mu=10.0,
        ),
        # Caps core-range L/D instead of maximizing it unboundedly: pushing L/D too
        # far degrades pitching-moment behavior and breaks OCP controllability
        # (observed as IPOPT solve failures / blown-up terminal error past ~5.5).
        Constraint(
            name="ld_cap",
            fn=lambda aero, alpha, re: masked_mean(aero["CL"] / aero["CD"], alpha, CORE_AOA_LO, CORE_AOA_HI),
            kind="le",
            target=5.0,
            mu=5.0,
        ),
    ]

    problem = AirfoilDesignProblem(
        device=device,
        objective=objective,
        constraints=constraints,
        model_size="xxxlarge",
        batch_side=8,
        lr=1e-4,
    )

    problem.train(n_iters=3000, log_every=100)

    airfoil = problem.to_kulfan_airfoil(name="Cl/Cd")

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    airfoil.draw()
    plt.show()

    print("upper", problem.vars.upper.detach().cpu().numpy())
    print("lower", problem.vars.lower.detach().cpu().numpy())
    print("LE", problem.vars.LE.detach().cpu().numpy())
    print("TE", problem.vars.TE.detach().cpu().numpy())

    config_path = Path(__file__).resolve().parents[1] / "conf" / "perching_2D.yaml"
    result = solve_ocp_once(problem, config_path)
    print(f"total_obj = {result['total_obj']}")

