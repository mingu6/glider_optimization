"""Assemble one task's finished co-design run into experiments/<task>/.

Copies the summary figure (from plot_task_figure.py) and trajectory set (from
make_task_trajectories.py) that already live under results/figures/, and writes
a small design.json with the final Kulfan parameters and key metrics pulled
straight from wandb -- so experiments/ is a self-contained record of what each
run actually produced, not a pointer back into results/ or wandb.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import wandb

ROOT = Path("/home/affinita/PhD/glider_optimization")

OBJECTIVE_KEYS = {
    "glide": ("distance_weight", None),
}


def latest_run(run_name):
    api = wandb.Api()
    runs = sorted([r for r in api.runs("glider-optimization") if r.name == run_name],
                  key=lambda r: r.created_at)
    if not runs:
        raise SystemExit(f"no wandb run named {run_name!r}")
    return runs[-1]


def best_row(h2):
    """The logged design with the lowest evaluation/objective_total (the raw,
    signed cost the optimiser actually minimises -- glide's is a negated
    distance, so "lowest" is uniformly "best" across tasks), falling back to
    the last row if that column isn't there. Using the literal last row is
    wrong for a run that was stopped early/manually: once the objective has
    plateaued, Adam with a decaying LR just noisily wanders +/-2% around it
    rather than monotonically improving, so the last iteration logged before
    a kill is not reliably the best one."""
    if "evaluation/objective_total" in h2.columns:
        sub = h2.dropna(subset=["evaluation/objective_total"])
        if not sub.empty:
            return sub.loc[sub["evaluation/objective_total"].idxmin()]
    return h2.iloc[-1]


def final_design_and_metrics(run_name, is_3d=False):
    run = latest_run(run_name)
    h = run.history(pandas=True)
    h2 = h.dropna(subset=["airfoil/upper_params_0"])
    last = best_row(h2)

    def vec(prefix, n=8):
        return [float(last[f"{prefix}_{i}"]) for i in range(n)]

    design = {
        "upper": vec("airfoil/upper_params"), "lower": vec("airfoil/lower_params"),
        "leading_edge_weight": float(last["airfoil/leading_edge_weight"]),
        "TE_thickness": float(last["airfoil/TE_thickness"]),
    }
    if is_3d:
        design.update({
            "upper_tip": vec("airfoil/upper_params_tip"), "lower_tip": vec("airfoil/lower_params_tip"),
            "leading_edge_weight_tip": float(last["airfoil/leading_edge_weight_tip"]),
            "TE_thickness_tip": float(last["airfoil/TE_thickness_tip"]),
        })

    metric_cols = [c for c in h.columns if any(
        k in c for k in ("objective", "cost", "primary_cost", "augmented_lagrangian", "confidence"))]
    at_best_step = h[h["_step"] == last["_step"]]
    metrics = {} if at_best_step.empty else {
        k: (None if (v := at_best_step.iloc[0][k]) != v else float(v))  # NaN -> None
        for k in metric_cols if k in at_best_step.columns
    }

    return {
        "wandb_run_id": run.id, "wandb_run_name": run.name, "wandb_url": run.url,
        "state": run.state, "best_step": int(last["_step"]), "last_logged_step": int(h2["_step"].max()),
        "n_logged_steps": int(len(h)),
        "design": design, "metrics_at_best_step": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="experiments/<task> folder name, e.g. landing, perching, glide, perching_3d")
    ap.add_argument("--run", required=True, help="wandb run name")
    ap.add_argument("--traj-tag", default=None, help="results/figures/traj_<tag>_{opt,seed} to copy, if any")
    ap.add_argument("--summary", default=None, help="results/figures/<summary> stem to copy (no extension)")
    ap.add_argument("--3d", dest="is_3d", action="store_true")
    args = ap.parse_args()

    out = ROOT / "experiments" / args.task
    out.mkdir(parents=True, exist_ok=True)

    info = final_design_and_metrics(args.run, is_3d=args.is_3d)
    (out / "design.json").write_text(json.dumps(info, indent=2))
    print(f"  wrote {out/'design.json'}  (run {info['wandb_run_name']!r}, "
          f"state={info['state']}, best step {info['best_step']} (last logged: {info['last_logged_step']}))")

    if args.summary:
        for ext in ("png", "pdf"):
            src = ROOT / f"results/figures/{args.summary}.{ext}"
            if src.exists():
                shutil.copy(src, out / f"summary.{ext}")
                print(f"  copied {src.name} -> {out}/summary.{ext}")

    if args.traj_tag:
        for which in ("opt", "seed"):
            src = ROOT / f"results/figures/traj_{args.traj_tag}_{which}"
            if src.is_dir():
                dst = out / f"trajectories_{which}"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"  copied {src} -> {dst} ({len(list(dst.glob('*.npy')))} files)")


if __name__ == "__main__":
    main()
