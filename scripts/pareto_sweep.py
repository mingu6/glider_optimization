"""Sweep the epsilon-constraint bound to trace the Pareto front.

The co-design problem is

    min_psi  Phi_1(psi)                     (primary task)
    s.t.     Phi_j(psi) <= eps_j,  j >= 2   (secondary tasks, as requirements)
             g(psi) <= 0                    (design constraints)

Each run pins eps and optimises to convergence, giving one Pareto point. Sweeping
eps walks the front. Unlike a weighted sum of the task costs this needs no
normalisation between tasks -- each bound is stated in its own task's units -- and
it recovers points on non-convex stretches of the front, which a weighted sum
provably cannot reach whatever the weights.

Usage:
    python scripts/pareto_sweep.py --config conf/pareto_2D.yaml \\
        --epsilons 8 10 12 14 16 --task landing [--iters 30] [--dry-run]

Each run writes its own wandb run named <prefix>_eps<value> and executes in its own
working directory, since the OCP block dumps per-iteration trajectories under names
that collide between runs. With --parallel the whole sweep runs at once, which suits
a many-core box: one run only needs an IPOPT worker per (task, launch condition).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Trace the Pareto front by sweeping epsilon")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--epsilons", type=float, nargs="+", required=True)
    p.add_argument("--task", type=str, default=None,
                   help="which secondary task's bound to sweep (default: the only one)")
    p.add_argument("--iters", type=int, default=None, help="override max_outer_iters")
    p.add_argument("--prefix", type=str, default=None, help="wandb run-name prefix")
    p.add_argument("--out-dir", type=Path, default=Path("runs/pareto"))
    p.add_argument("--dry-run", action="store_true", help="print what would run, then stop")
    p.add_argument("--parallel", action="store_true",
                   help="run every epsilon concurrently. Each run only needs one IPOPT "
                        "worker per (task, launch condition), so a many-core box can "
                        "hold the whole sweep at once instead of serialising it.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base = yaml.safe_load(args.config.read_text())
    tasks = base.get("tasks") or []
    if len(tasks) < 2:
        raise SystemExit(f"{args.config} has no secondary task to constrain")

    secondary = [t["name"] for t in tasks[1:]]
    task_name = args.task or (secondary[0] if len(secondary) == 1 else None)
    if task_name is None:
        raise SystemExit(f"--task is required; config has several: {secondary}")
    if task_name not in secondary:
        raise SystemExit(f"'{task_name}' is not a secondary task ({secondary})")

    prefix = args.prefix or args.config.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index, running = [], []

    for eps in args.epsilons:
        cfg = yaml.safe_load(args.config.read_text())
        for task in cfg["tasks"]:
            if task["name"] == task_name:
                task["epsilon"] = float(eps)
        if args.iters is not None:
            cfg["run"]["max_outer_iters"] = args.iters

        tag = f"{prefix}_eps{eps:g}"
        cfg["io"]["checkpoint_dir"] = str(args.out_dir / tag)
        cfg["io"].setdefault("wandb", {})
        cfg["io"]["wandb"]["tags"] = list(cfg["io"]["wandb"].get("tags", [])) + ["pareto", f"eps={eps:g}"]

        cfg_path = (args.out_dir / f"{tag}.yaml").resolve()
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        log_path = args.out_dir / f"{tag}.log"
        # own cwd: the OCP block saves traj_<task>_<i>_<iter>.npy relative to it, and
        # those names are identical across runs
        workdir = (args.out_dir / tag).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "glider_optimization.main",
               "--config", str(cfg_path), "--run-name", tag]
        index.append({"epsilon": eps, "run_name": tag, "config": str(cfg_path), "log": str(log_path)})

        print(f"[pareto] eps({task_name}) = {eps:g}  ->  {tag}")
        if args.dry_run:
            continue

        log = log_path.open("w")
        if args.parallel:
            running.append((tag, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                  cwd=workdir), log))
        else:
            with log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=workdir)
            if result.returncode != 0:
                # One infeasible epsilon should not abandon the rest of the front.
                print(f"[pareto] WARNING: {tag} exited {result.returncode}; see {log_path}")

    for tag, proc, log in running:
        rc = proc.wait()
        log.close()
        if rc != 0:
            print(f"[pareto] WARNING: {tag} exited {rc}")

    (args.out_dir / f"{prefix}_index.json").write_text(json.dumps(index, indent=2))
    print(f"[pareto] wrote {args.out_dir / (prefix + '_index.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
