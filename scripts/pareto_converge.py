"""Resume each front point's epsilon until its margin closes.

A Pareto point needs its bound active: Phi_2 == epsilon. The front as first produced
has wide margins (+9.8 to +14.0), meaning those links stopped in the interior with
glide still on the table. Each round here re-runs one epsilon seeded from that
epsilon's own most recent design, so the optimisation continues rather than restarts,
and stops when the margin is within tolerance (or when it stops improving, which
means more rounds will not help).
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/affinita/PhD/glider_optimization")


def last_point(front_path):
    if not front_path.exists():
        return None
    pts = json.loads(front_path.read_text())
    return pts[-1] if pts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=float, nargs="+", default=[215.7, 216.5, 217.3])
    ap.add_argument("--tol", type=float, default=0.5, help="|margin| considered active")
    ap.add_argument("--iters", type=int, default=20, help="iterations per round")
    ap.add_argument("--rounds", type=int, default=12, help="max rounds per epsilon")
    ap.add_argument("--out", default="runs/converge")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("converge")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for eps in args.epsilons:
        tag = f"eps{int(eps)}"
        eps_dir = out / tag
        eps_dir.mkdir(parents=True, exist_ok=True)
        # carry the landing basins in; without them Phi_2 is not comparable round to round
        src = ROOT / "runs/pareto_eps9/warm_start.pkl"
        if src.exists() and not (eps_dir / "warm_start.pkl").exists():
            (eps_dir / "warm_start.pkl").write_bytes(src.read_bytes())

        best = None
        prev_margin = None
        for rnd in range(1, args.rounds + 1):
            # seed from this epsilon's own latest run, so each round continues the last
            cmd = [sys.executable, str(ROOT / "scripts/pareto_chain.py"),
                   "--epsilons", str(eps), "--iters", str(args.iters),
                   "--warmup-iters", "0", "--link-lr", "0.0015", "--link-gamma", "0.999",
                   "--seed-run", f"pchain_{tag}", "--out", str(eps_dir.relative_to(ROOT))]
            with open(eps_dir / f"round{rnd}.log", "w") as fh:
                subprocess.call(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)

            pt = last_point(eps_dir / "front.json")
            if pt is None:
                log.error(f"{tag} round {rnd}: no point produced; stopping this epsilon")
                break
            margin = pt["margin"]
            log.info(f"{tag} round {rnd}: glide={pt['glide_m']:.2f} m  "
                     f"Phi_2={pt['landing_cost']:.2f}  margin={margin:+.2f}")
            if best is None or abs(margin) < abs(best["margin"]):
                best = pt

            if abs(margin) <= args.tol:
                log.info(f"{tag}: margin active within {args.tol}; done in {rnd} rounds")
                break
            # Do NOT stop on a non-improving margin. The design descends smoothly
            # (thickness falling ~0.001/iteration) while the measured objectives swing
            # by several metres, because the glide OCP hops between local optima. The
            # margin therefore stays wide, or widens, until the design crosses into the
            # thin/high-glide basin -- at which point it closes abruptly. Stopping on a
            # flat margin would abort exactly that traverse. Stop only if the design
            # itself stops moving.
            if prev_margin is not None and abs(margin - prev_margin) < 1e-6:
                log.info(f"{tag}: margin identical between rounds; design is static")
                break
            prev_margin = margin

        summary[tag] = best
        (out / "converged.json").write_text(json.dumps(summary, indent=2))

    log.info("=== convergence summary ===")
    for tag, pt in summary.items():
        if pt:
            log.info(f"  {tag}: glide={pt['glide_m']:.2f} m  Phi_2={pt['landing_cost']:.2f}  "
                     f"margin={pt['margin']:+.2f}")


if __name__ == "__main__":
    main()
