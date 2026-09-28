"""Energy audit: does any accepted trajectory gain energy out of nothing?

The glide run produced a trajectory that climbed from 5.8 m to 242 m of specific
energy and back, travelling 174 m while the other three launches covered 58-79 m.
IPOPT reported that solve as successful, so nothing downstream rejected it, and the
mean objective was inflated by an artefact rather than by a better wing.

In still air the specific energy E = z + v^2 / 2g can only decrease. This
re-evaluates a stored design with the design frozen, then audits every resulting
trajectory against that bound, so a task can be cleared (or not) before its numbers
go into a paper.
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import wandb

from glider_optimization.config import load_config
from glider_optimization.runner import Runner

ROOT = Path("/home/affinita/PhD/glider_optimization")
G = 9.81

TASKS = {
    "landing":  ("conf/best_landing.yaml",  "best_landing",  "softlanding"),
    "perching": ("conf/best_perching.yaml", "best_perching", "perching"),
    "glide":    ("conf/best_glide.yaml",    "best_glide",    "distance"),
}


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


def audit(task, tol=0.05):
    conf, run_name, traj_tag = TASKS[task]
    for p in ROOT.glob(f"traj_{traj_tag}_*_*.npy"):
        p.unlink()

    cfg = load_config(ROOT / conf)
    cfg.io.wandb.enabled = False
    cfg.run.max_outer_iters = 1
    cfg.airfoil.lr = 0.0                     # frozen: reproduce, do not optimise
    cfg.io.checkpoint_dir = f"./runs/audit_{task}"
    cfg.ocp.warm_start_path = None
    d = design_from_wandb(run_name)
    cfg.airfoil.upper_initial_weights = d["upper"]
    cfg.airfoil.lower_initial_weights = d["lower"]
    cfg.airfoil.leading_edge_weight = d["le"]
    cfg.airfoil.TE_thickness = d["te"]

    runner = Runner(cfg)
    runner.run()

    print(f"\n=== {task}: specific energy audit (E = z + v^2/2g, must not increase)")
    print(f"{'launch':>6} | {'E(0)':>8} | {'E(max)':>8} | {'gain':>9} | {'x_end':>8} | verdict")
    bad = []
    for i, traj in enumerate(runner.blocks["OCP"].last_trajs):
        if not traj["success"]:
            print(f"{i:>6} |        - |        - |         - |        - | SOLVE FAILED")
            continue
        st = traj["state_traj_opt"]
        E = st[:, 1] + (st[:, 4] ** 2 + st[:, 5] ** 2) / (2 * G)
        gain = float(E.max() - E[0])
        ok = gain <= tol
        if not ok:
            bad.append(i)
        print(f"{i:>6} | {E[0]:8.3f} | {E.max():8.3f} | {gain:+9.3f} | {st[-1,0]:8.2f} | "
              f"{'ok' if ok else 'ENERGY GAIN - trajectory is not dynamically feasible'}")
    print(f"  verdict: {'PASS' if not bad else f'FAIL on launches {bad}'}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=["landing", "perching"],
                    choices=list(TASKS) + [])
    ap.add_argument("--tol", type=float, default=0.05, help="allowed gain [m]")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    for t in args.tasks:
        audit(t, args.tol)


if __name__ == "__main__":
    main()
