"""Re-derive the epsilon band from converged designs, not from brief settles.

The first band came from anchors settled for 8 iterations. That was not long
enough for every landing OCP to leave a bad local optimum: launch z=3 sat at a
cost of ~350 for the first ~80 iterations of the sweep and then dropped to ~216
in a single step, which alone shifted Phi_2 by ~33. Both anchors were inflated by
roughly that much, so every epsilon in the resulting band came out slack.

This measures each anchor with a long settle from a shared warm-start reservoir,
and reports the per-launch spread so a launch still stuck in a bad basin is
visible rather than silently averaged in.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

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


def design_from_npz(path):
    d = np.load(path)
    return {"upper": d["upper"], "lower": d["lower"],
            "le": float(d["le"]), "te": float(d["te"])}


def evaluate(label, design, settle, reservoir, out):
    cfg = load_config(CONFIG)
    cfg.io.wandb.enabled = False
    cfg.run.max_outer_iters = settle
    cfg.airfoil.lr = 0.0                       # hold the design; only the OCPs settle
    cfg.io.checkpoint_dir = str(out / f"cal_{label}")
    work = out / f"cal_{label}" / "warm.pkl"
    work.parent.mkdir(parents=True, exist_ok=True)
    if reservoir and Path(reservoir).exists():
        shutil.copy(reservoir, work)           # start every anchor from the same basins
    cfg.ocp.warm_start_path = str(work)
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
    spread = (max(per) - min(per)) if per else float("nan")
    print(f"  {label:18s} glide {glide:7.2f} m   Phi_2 {landing:8.2f}   "
          f"per-launch {[round(p, 1) for p in per]}  spread {spread:.1f}", flush=True)
    return glide, landing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=int, default=60,
                    help="iterations to settle each anchor (8 was far too few)")
    ap.add_argument("--reservoir", default="runs/pareto_v5/warm_start.pkl")
    ap.add_argument("--glide-design", default=None,
                    help="npz with the glide-optimal design; omit to use --glide-run")
    ap.add_argument("--glide-run", default="pchain_eps262",
                    help="wandb run whose final design is the glide-optimal anchor")
    ap.add_argument("--landing-run", default="softlanding_fixed")
    ap.add_argument("--out", default="runs/calibration")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    reservoir = ROOT / args.reservoir

    print(f"Calibrating with {args.settle} settle iterations, shared reservoir:", flush=True)
    glide_design = (design_from_npz(ROOT / args.glide_design) if args.glide_design
                    else design_from_wandb(args.glide_run))
    g_hi, phi2_loose = evaluate("glide-optimal", glide_design, args.settle, reservoir, out)
    land_design = design_from_wandb(args.landing_run)
    g_lo, phi2_tight = evaluate("landing-optimal", land_design, args.settle, reservoir, out)

    result = {"glide_optimal": {"glide_m": g_hi, "landing_cost": phi2_loose},
              "landing_optimal": {"glide_m": g_lo, "landing_cost": phi2_tight}}
    if phi2_loose > phi2_tight:
        span = phi2_loose - phi2_tight
        result["epsilons"] = [round(phi2_tight + f * span, 2)
                              for f in (0.85, 0.68, 0.50, 0.32, 0.15)]
        print(f"\n  band ({phi2_tight:.1f}, {phi2_loose:.1f}) -> {result['epsilons']}")
    else:
        result["epsilons"] = []
        print("\n  NO BAND: Phi_2 at the glide-optimal design is not above the landing-optimal one")
    (out / "band.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
