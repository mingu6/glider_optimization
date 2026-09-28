"""Evaluate the epsilon-constraint designs under one identical procedure.

The epsilon-constraint chain generates the designs -- one per epsilon -- but the
Phi values it reports along the way are not comparable between links: they depend
on how fresh each link's warm-start reservoir is. Measured drift of 16 units in ten
iterations, against a band only 2.4 units wide, makes the in-run numbers unusable
as front coordinates even though they are fine as optimisation feedback.

So the front is plotted from a second pass: every design evaluated here, with the
same reservoir, the same settle count and the design frozen, so the only thing that
differs between points is the wing. Per-launch costs are reported for each so a
landing sitting in a bad local optimum is visible rather than silently averaged in
(cost should increase with launch height).
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


def design_from_npz(path):
    d = np.load(path)
    return {"upper": d["upper"], "lower": d["lower"],
            "le": float(d["le"]), "te": float(d["te"])}


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
    cfg.airfoil.lr = 0.0                        # frozen: measuring, not optimising
    cfg.io.checkpoint_dir = str(out / tag)
    work = out / tag / "warm.pkl"
    work.parent.mkdir(parents=True, exist_ok=True)
    if reservoir.exists():
        shutil.copy(reservoir, work)            # identical starting basins for every design
    cfg.ocp.warm_start_path = str(work)
    for t in cfg.task_list:
        t.epsilon = None                        # evaluation only, no constraint
    cfg.airfoil.upper_initial_weights = design["upper"]
    cfg.airfoil.lower_initial_weights = design["lower"]
    cfg.airfoil.leading_edge_weight = design["le"]
    cfg.airfoil.TE_thickness = design["te"]

    runner = Runner(cfg)
    runner.run()
    costs = runner.blocks["Evaluation"].task_costs
    per = [float(t["cost"][0][0]) for t in runner.blocks["OCP"].trajs_by_task["softlanding"]
           if t["success"]]
    monotone = all(per[i] <= per[i + 1] for i in range(len(per) - 1)) if len(per) > 1 else True
    tc, camber = geometry(design)
    row = {"tag": tag, "t_over_c": tc, "camber": camber,
           "glide_m": -costs["distance"], "landing_cost": costs["softlanding"],
           "per_launch": per, "monotone": monotone}
    print(f"  {tag:16s} t/c {tc:.4f}  glide {row['glide_m']:7.2f} m  "
          f"Phi_2 {row['landing_cost']:8.2f}  {'ok' if monotone else 'BAD-BASIN'}", flush=True)
    return row


def non_dominated(points):
    """Maximise glide, minimise landing cost."""
    out = []
    for p in points:
        if any(q is not p and q["glide_m"] >= p["glide_m"] and q["landing_cost"] <= p["landing_cost"]
               and (q["glide_m"] > p["glide_m"] or q["landing_cost"] < p["landing_cost"])
               for q in points):
            continue
        out.append(p["tag"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", default="runs/pareto_eps3")
    ap.add_argument("--settle", type=int, default=6)
    ap.add_argument("--reservoir", default="runs/pareto_final2/warm_start.pkl")
    ap.add_argument("--extra-runs", nargs="*", default=["softlanding_fixed", "pchain_eps262"],
                    help="wandb runs to include as reference designs")
    ap.add_argument("--out", default="runs/front_final")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    reservoir = ROOT / args.reservoir

    designs = []
    for npz in sorted((ROOT / args.chain_dir).glob("design_eps*.npz")):
        designs.append((npz.stem.replace("design_", ""), design_from_npz(npz)))
    for name in args.extra_runs:
        try:
            designs.append((name, design_from_wandb(name)))
        except Exception as exc:
            print(f"  (skipping {name}: {exc})", flush=True)

    if not designs:
        print("no designs found -- has the chain produced any completed links?")
        return 1

    print(f"Evaluating {len(designs)} designs, {args.settle} settles, shared reservoir:",
          flush=True)
    rows = []
    for tag, d in designs:
        rows.append(evaluate(tag, d, args.settle, reservoir, out))
        (out / "front.json").write_text(json.dumps(
            {"points": rows, "non_dominated": non_dominated(rows)}, indent=2))

    front = non_dominated(rows)
    print(f"\n  non-dominated: {front}")
    print(f"\n  {'design':16s} {'t/c':>7} {'glide (m)':>10} {'Phi_2':>9}  front")
    for r in sorted(rows, key=lambda r: r["glide_m"]):
        print(f"  {r['tag']:16s} {r['t_over_c']:7.4f} {r['glide_m']:10.2f} "
              f"{r['landing_cost']:9.2f}  {'YES' if r['tag'] in front else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
