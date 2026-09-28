"""Regenerate trajectories for each task's optimised design and file them per task.

The runner writes traj_<task>_<launch>_<iter>.npy into the working directory, which
different runs overwrite. This evaluates one frozen design per task and moves the
result into results/figures/traj_<task>/ so the plotting script has a stable, clearly
attributed set -- and so the energy audit that cleared them applies to exactly the
trajectories that get plotted.
"""
import logging, shutil, sys
from pathlib import Path
import numpy as np, wandb
from glider_optimization.config import load_config
from glider_optimization.runner import Runner

ROOT = Path("/home/affinita/PhD/glider_optimization")
G = 9.81
TASKS = {
    "landing":     ("conf/best_landing.yaml",  "best_landing",  "softlanding", False),
    "perching":    ("conf/best_perching.yaml", "best_perching", "perching",    False),
    "glide":       ("conf/best_glide.yaml",    "best_glide",    "distance",    False),
    "perching_3d": ("conf/perching_3D.yaml",   "perching3D",    "perching",    True),
}


def design_from_wandb(run_name, is_3d=False):
    api = wandb.Api()
    r = sorted([q for q in api.runs("glider-optimization") if q.name == run_name],
               key=lambda q: q.created_at)[-1]
    h2 = r.history(pandas=True).dropna(subset=["airfoil/upper_params_0"])
    # Best-by-objective, not literally-last: a run stopped early (or one whose LR has
    # decayed to where Adam just noisily wanders) doesn't have its best design in the
    # last logged row -- see collect_experiment.py's best_row for the full rationale.
    if "evaluation/objective_total" in h2.columns:
        sub = h2.dropna(subset=["evaluation/objective_total"])
        row = sub.loc[sub["evaluation/objective_total"].idxmin()] if not sub.empty else h2.iloc[-1]
    else:
        row = h2.iloc[-1]
    d = {"upper": np.array([float(row[f"airfoil/upper_params_{i}"]) for i in range(8)]),
         "lower": np.array([float(row[f"airfoil/lower_params_{i}"]) for i in range(8)]),
         "le": float(row["airfoil/leading_edge_weight"]),
         "te": float(row["airfoil/TE_thickness"])}
    if is_3d:
        d.update({
            "upper_tip": np.array([float(row[f"airfoil/upper_params_tip_{i}"]) for i in range(8)]),
            "lower_tip": np.array([float(row[f"airfoil/lower_params_tip_{i}"]) for i in range(8)]),
            "le_tip": float(row["airfoil/leading_edge_weight_tip"]),
            "te_tip": float(row["airfoil/TE_thickness_tip"]),
        })
    return d


def generate(task, seed_design=False):
    conf, run_name, tag, is_3d = TASKS[task]
    # Scoped to this task's own tag, not traj_*_*.npy generally: two tasks run
    # concurrently (e.g. from separate invocations) would otherwise race on each
    # other's in-progress output, since the runner writes into this same ROOT dir
    # regardless of which script instance started it.
    for p in ROOT.glob(f"traj_{tag}_*_*.npy"):
        p.unlink()
    cfg = load_config(ROOT / conf)
    cfg.io.wandb.enabled = False
    cfg.run.max_outer_iters = 1
    cfg.airfoil.lr = 0.0
    cfg.ocp.warm_start_path = None          # cold: reproducible, and audited clean
    suffix = "seed" if seed_design else "opt"
    cfg.io.checkpoint_dir = f"./runs/trajgen_{task}_{suffix}"
    d = None if seed_design else design_from_wandb(run_name, is_3d=is_3d)
    if d is not None:
        cfg.airfoil.upper_initial_weights = d["upper"]
        cfg.airfoil.lower_initial_weights = d["lower"]
        cfg.airfoil.leading_edge_weight = d["le"]
        cfg.airfoil.TE_thickness = d["te"]
    runner = Runner(cfg)
    if d is not None and is_3d:
        # Airfoil3D.__init__ seeds both root and tip from the same config field (the
        # yaml only specifies one shape; they diverge through training), so the tip's
        # trained shape has to be injected after construction -- there's no config path
        # for two different initial airfoils.
        import torch
        af = runner.blocks["Airfoil"]
        af.upper_params_tip.data = torch.tensor(d["upper_tip"], dtype=torch.float32)
        af.lower_params_tip.data = torch.tensor(d["lower_tip"], dtype=torch.float32)
        af.leading_edge_param_tip.data = torch.tensor([d["le_tip"]], dtype=torch.float32)
        af.TE_thickness_param_tip.data = torch.tensor([d["te_tip"]], dtype=torch.float32)
    runner.run()

    out = ROOT / f"results/figures/traj_{task}_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.npy"):
        p.unlink()
    rejected_dir = out.parent / f"traj_{task}_{suffix}_rejected"
    if rejected_dir.exists():
        for p in rejected_dir.glob("*.npy"):
            p.unlink()
    n_bad = 0
    for p in ROOT.glob(f"traj_{tag}_*_*.npy"):
        st = np.load(p)
        e = st[:, 1] + (st[:, 4] ** 2 + st[:, 5] ** 2) / (2 * G)
        gain = float(np.max(e) - e[0])
        if gain > 0.05:
            # OCP.forward() writes state_traj_opt unconditionally, even for a solve it
            # has itself rejected as dynamically infeasible (energy_gain_tol) -- so this
            # file existing is not evidence it's a real result. Don't let it into the
            # plotted set silently; keep it aside instead of just deleting it.
            n_bad += 1
            rejected_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, rejected_dir / p.name)
            print(f"  WARNING: {p.name} rejected: specific energy rose {gain:.2f} m "
                  f"(tol 0.05) -- dynamically infeasible, moved to {rejected_dir.name}/ instead of {out.name}/",
                  flush=True)
            continue
        shutil.copy(p, out / p.name)
    print(f"  {task:9s} {suffix}: {len(list(out.glob('*.npy')))} trajectories -> {out.name}"
          f"{f'  ({n_bad} ENERGY-GAINING, excluded)' if n_bad else '  (all energy-conserving)'}",
          flush=True)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    for t in (sys.argv[1:] or ["landing", "perching", "glide"]):
        generate(t, seed_design=False)
        generate(t, seed_design=True)
