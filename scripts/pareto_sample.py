"""Pareto front by direct evaluation of a design family, not by epsilon-constraint.

The epsilon-constraint sweep cannot work here: measured under one consistent
procedure, Phi_2 (SoftLanding cost) spans only 215.4 -> 217.8 between the
landing-optimal and glide-optimal designs, while Phi_1 (glide distance) spans
16 -> 44 m. A 2.4-unit band is smaller than the solver's own scatter, so no
epsilon can bind meaningfully.

Instead: walk a family of designs from landing-optimal to glide-optimal, evaluate
BOTH objectives for each under an identical procedure and a shared good-basin
reservoir, and report the non-dominated subset. This is an a-posteriori Pareto
approximation -- weaker than a solved front in that the family may not contain the
true optimum for any weighting, but every point is a real, comparably-measured
design rather than an artefact of where an optimiser happened to stop.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import aerosandbox as asb
import numpy as np
import wandb

from glider_optimization.config import load_config
from glider_optimization.runner import Runner

ROOT = Path("/home/affinita/PhD/glider_optimization")
CONFIG = ROOT / "conf/pareto_soft_2D.yaml"


def design_from_wandb(run_name):
    api = wandb.Api()
    run = sorted([r for r in api.runs("glider-optimization") if r.name == run_name],
                 key=lambda r: r.created_at)[-1]
    row = run.history(pandas=True).dropna(subset=["airfoil/upper_params_0"]).iloc[-1]
    return {
        "upper": np.array([float(row[f"airfoil/upper_params_{i}"]) for i in range(8)]),
        "lower": np.array([float(row[f"airfoil/lower_params_{i}"]) for i in range(8)]),
        "le": float(row["airfoil/leading_edge_weight"]),
        "te": float(row["airfoil/TE_thickness"]),
    }


def blend(a, b, t):
    return {k: (1 - t) * a[k] + t * b[k] for k in ("upper", "lower", "le", "te")}


def geometry(design):
    x = np.linspace(0, 1, 300)
    foil = asb.KulfanAirfoil(name="a", upper_weights=design["upper"],
                             lower_weights=design["lower"],
                             leading_edge_weight=design["le"],
                             TE_thickness=design["te"])
    return float(np.max(foil.local_thickness(x))), float(np.max(np.abs(foil.local_camber(x))))


def evaluate(tag, design, settle, reservoir, out):
    cfg = load_config(CONFIG)
    cfg.io.wandb.enabled = False
    cfg.run.max_outer_iters = settle
    cfg.airfoil.lr = 0.0                       # design frozen: measure, do not optimise
    cfg.io.checkpoint_dir = str(out / tag)
    work = out / tag / "warm.pkl"
    work.parent.mkdir(parents=True, exist_ok=True)
    if reservoir.exists():
        shutil.copy(reservoir, work)           # same basins for every design
    cfg.ocp.warm_start_path = str(work)
    for t in cfg.task_list:
        t.epsilon = None                       # pure evaluation, no constraint
    cfg.airfoil.upper_initial_weights = design["upper"]
    cfg.airfoil.lower_initial_weights = design["lower"]
    cfg.airfoil.leading_edge_weight = design["le"]
    cfg.airfoil.TE_thickness = design["te"]

    runner = Runner(cfg)
    runner.run()
    costs = runner.blocks["Evaluation"].task_costs
    glide, landing = -costs["distance"], costs["softlanding"]
    per = [float(t["cost"][0][0]) for t in runner.blocks["OCP"].trajs_by_task["softlanding"]
           if t["success"]]
    # cost should rise with launch height; if it does not, a landing sits in a bad
    # basin and this point is not comparable to the others
    healthy = all(per[i] <= per[i + 1] for i in range(len(per) - 1)) if len(per) > 1 else True
    tc, camber = geometry(design)
    print(f"  {tag:14s} t/c {tc:.4f}  camber {camber:.4f}  glide {glide:7.2f} m  "
          f"Phi_2 {landing:8.2f}  {'ok' if healthy else 'BAD-BASIN'}", flush=True)
    return {"tag": tag, "t_over_c": tc, "camber": camber, "glide_m": glide,
            "landing_cost": landing, "per_launch": per, "monotone": healthy}


def non_dominated(points):
    """Maximise glide, minimise landing cost."""
    keep = []
    for p in points:
        if p["glide_m"] is None or p["landing_cost"] is None:
            continue
        dominated = any(q is not p
                        and q["glide_m"] >= p["glide_m"]
                        and q["landing_cost"] <= p["landing_cost"]
                        and (q["glide_m"] > p["glide_m"] or q["landing_cost"] < p["landing_cost"])
                        for q in points)
        if not dominated:
            keep.append(p["tag"])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=7, help="designs between the two extremes")
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--reservoir", default="runs/pareto_final2/warm_start.pkl")
    ap.add_argument("--landing-run", default="softlanding_fixed")
    ap.add_argument("--glide-run", default="pchain_eps262")
    ap.add_argument("--out", default="runs/pareto_sampled")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    reservoir = ROOT / args.reservoir

    land = design_from_wandb(args.landing_run)
    glide = design_from_wandb(args.glide_run)

    print(f"Sampling {args.steps + 2} designs from landing-optimal to glide-optimal:",
          flush=True)
    results = []
    for i, t in enumerate(np.linspace(0.0, 1.0, args.steps + 2)):
        d = blend(land, glide, float(t))
        results.append(evaluate(f"t{t:.2f}", d, args.settle, reservoir, out))
        (out / "sampled.json").write_text(json.dumps(results, indent=2))

    front = non_dominated(results)
    (out / "sampled.json").write_text(json.dumps(
        {"points": results, "non_dominated": front}, indent=2))
    print(f"\n  non-dominated: {front}")


if __name__ == "__main__":
    main()
