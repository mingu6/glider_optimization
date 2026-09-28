"""Trace the Pareto front by continuation: each tighter epsilon starts from the
previous link's design *and* its warm-start trajectories.

Independent per-epsilon runs do not produce a front. The AL penalty is one-sided,
so once Phi_2 falls below epsilon the multiplier decays, the gradient vanishes and
the run quietly becomes an unconstrained Phi_1 maximisation that ends in whatever
basin it wandered into -- points come out non-monotone in epsilon and can dominate
one another.

Two things must carry between links. The design is obvious; the OCP warm-start
reservoir matters just as much, because a cold solve of the landing task reports
~365 where a warm one reports ~238. A link that starts cold shows the AL a
violation of ~135 that does not exist, spikes lambda and rho, and drives the
design into the ground before the reservoir refills.

Each link runs as its own process (the in-process version deadlocked in
multiprocessing), handing state forward through files: the design as npz, the
reservoir as the pickle at ocp.warm_start_path. That also makes the chain
restartable -- rerun with --start-at to resume from a completed link.
"""

import argparse
import json
import os
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import wandb
import yaml

ROOT = Path("/home/affinita/PhD/glider_optimization")
BASE_CONFIG = ROOT / "conf/pareto_soft_2D.yaml"


def seed_from_wandb(run_name):
    api = wandb.Api()
    run = sorted([r for r in api.runs("glider-optimization") if r.name == run_name],
                 key=lambda r: r.created_at)[-1]
    row = run.history(pandas=True).dropna(subset=["airfoil/upper_params_0"]).iloc[-1]
    return {
        "upper": [float(row[f"airfoil/upper_params_{i}"]) for i in range(8)],
        "lower": [float(row[f"airfoil/lower_params_{i}"]) for i in range(8)],
        "le": float(row["airfoil/leading_edge_weight"]),
        "te": float(row["airfoil/TE_thickness"]),
    }


def design_from_wandb(run_name):
    return seed_from_wandb(run_name)


def costs_from_log(log_path):
    """Last logged (glide distance, landing cost) from a link's stdout."""
    import re
    d = l = None
    for line in Path(log_path).read_text().splitlines():
        m = re.search(r"distance=(-?[\d.]+), softlanding=([\d.]+)", line)
        if m:
            d, l = -float(m.group(1)), float(m.group(2))
    return d, l


def run_link(name, eps, iters, design, warm_path, out_dir, log, lr, gamma):
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg["run"]["max_outer_iters"] = iters
    cfg["io"]["checkpoint_dir"] = str(out_dir / name)
    cfg["io"]["wandb"]["tags"] = ["ParetoChain"]
    # Log every iteration: the front is read back from the last logged objective line,
    # and with log_every > iters only iteration 0 is ever written -- which silently
    # recorded each link's *starting* state as its result.
    cfg["io"]["log_every"] = 1
    cfg["ocp"]["warm_start_path"] = str(warm_path)
    # Carry Adam moments, the LR schedule position and the AL duals between links too.
    # Adjacent epsilons are adjacent problems: lambda for the bound is nearly the same
    # from one link to the next, so restarting the duals (and Adam) each link made
    # every link re-derive what its predecessor already knew.
    cfg["run"]["chain_state_path"] = str(out_dir / "chain_state.pt")
    # The seed design comes from a run whose LR had already decayed; restarting each
    # link at the configured lr0 takes steps ~5x too large and knocks the converged
    # design out of its optimum (an unconstrained warm-up measurably *lost* glide
    # distance that way). Continuation wants small, steady steps instead.
    cfg["airfoil"]["lr"] = lr
    cfg["airfoil"]["gamma"] = gamma
    cfg["airfoil"]["upper_initial_weights"] = design["upper"]
    cfg["airfoil"]["lower_initial_weights"] = design["lower"]
    cfg["airfoil"]["leading_edge_weight"] = design["le"]
    cfg["airfoil"]["TE_thickness"] = design["te"]
    # stiffen the task constraint so it can hold the design against the glide
    # objective; at the shared rho the landing gradient is too weak to bind
    if os.environ.get("GLIDER_TASK_RHO"):
        cfg.setdefault("neuralFoilSampling", {})["task_rho"] = float(os.environ["GLIDER_TASK_RHO"])
    for t in cfg["tasks"]:
        if t["name"] == "softlanding":
            t["epsilon"] = float(eps)
            if os.environ.get("GLIDER_EPS_ACTIVE", "0") == "1":
                # drive Phi_2 onto the bound rather than merely under it
                t["epsilon_active"] = True

    cfg_path = out_dir / f"{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_path = out_dir / f"{name}.log"

    log.info(f"--- {name}: eps={eps}, {iters} iters -> {log_path}")
    with open(log_path, "w") as fh:
        rc = subprocess.call(
            [sys.executable, "-m", "glider_optimization.main",
             "--config", str(cfg_path), "--run-name", name],
            cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        log.error(f"--- {name} exited with code {rc}")
    return log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=float, nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=110)
    ap.add_argument("--warmup-iters", type=int, default=25)
    ap.add_argument("--seed-run", default="psoft_eps290")
    ap.add_argument("--seed-npz", default=None,
                    help="seed from a design npz instead of a wandb run; unambiguous "
                         "when several runs share a name")
    ap.add_argument("--out", default="runs/pareto_chain")
    ap.add_argument("--link-lr", type=float, default=0.002,
                    help="per-link learning rate (matches the decayed seed, not lr0)")
    ap.add_argument("--link-gamma", type=float, default=0.995)
    ap.add_argument("--start-at", type=float, default=None,
                    help="resume the chain at this epsilon (earlier links assumed done)")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("chain")

    warm_path = out / "warm_start.pkl"
    front_path = out / "front.json"
    results = json.loads(front_path.read_text()) if front_path.exists() else []

    if args.seed_npz:
        import numpy as _np
        _d = _np.load(ROOT / args.seed_npz)
        design = {"upper": _d["upper"].tolist(), "lower": _d["lower"].tolist(),
                  "le": float(_d["le"]), "te": float(_d["te"])}
        log.info(f"seeded from {args.seed_npz}")
    else:
        design = seed_from_wandb(args.seed_run)
        log.info(f"seeded from {args.seed_run}")

    # Unconstrained warm-up: fill the reservoir before any bound is applied, so the
    # first link is judged on warm solves like every later one.
    if args.warmup_iters > 0 and args.start_at is None:
        run_link("pchain_warmup", 1.0e9, args.warmup_iters, design, warm_path, out, log,
                 args.link_lr, args.link_gamma)
        design = design_from_wandb("pchain_warmup")
        d, l = costs_from_log(out / "pchain_warmup.log")
        log.info(f"--- warm-up done: glide={d:.2f} m, landing={l:.2f} (warm reference)")

    started = args.start_at is None
    for eps in args.epsilons:
        if not started:
            started = abs(eps - args.start_at) < 1e-9
            if not started:
                continue
        name = f"pchain_eps{int(eps)}"
        run_link(name, eps, args.iters, design, warm_path, out, log,
                 args.link_lr, args.link_gamma)
        try:
            design = design_from_wandb(name)
        except Exception as exc:
            log.error(f"--- {name}: could not read design back ({exc}); stopping")
            break
        np.savez(out / f"design_eps{int(eps)}.npz", **{k: np.array(v) for k, v in design.items()})
        d, l = costs_from_log(out / f"{name}.log")
        results.append({"epsilon": eps, "glide_m": d, "landing_cost": l,
                        "margin": None if l is None else eps - l})
        front_path.write_text(json.dumps(results, indent=2))
        log.info(f"--- {name} done: glide={d:.2f} m, landing={l:.2f}, margin={eps - l:+.2f}")

    log.info("chain complete")
    for r in results:
        log.info(f"  eps={r['epsilon']:6.1f}  glide={r['glide_m']:6.2f} m  "
                 f"landing={r['landing_cost']:7.2f}  margin={r['margin']:+7.2f}")


if __name__ == "__main__":
    main()
