"""Run the whole Pareto pipeline unattended: wait for anchors, pick the band, chain.

The epsilon band has to lie strictly between Phi_2 at the distance-optimal design
(loose end) and Phi_2 at the landing-optimal design (tight end). Outside it every
bound is slack or infeasible and the sweep produces nothing -- which is how the
first three attempts were lost. This waits for the measured anchors rather than
guessing, then runs the continuation chain from loose to tight.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/affinita/PhD/glider_optimization")
ANCHOR_LOG = Path("/tmp/claude-307861/-home-affinita-PhD-glider-optimization/"
                  "30475ee8-7bad-47dc-adec-31a4fc25e67e/scratchpad/mt/anchors.log")

ANCHOR_RE = re.compile(r"^\s*(\S+)\s+glide =\s*([\d.-]+) m\s+landing cost =\s*([\d.-]+)")


def read_anchors(path):
    """label -> (glide, landing cost), for whatever has been measured so far."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        m = ANCHOR_RE.match(line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def wait_for_anchors(path, needed, log, poll=120, timeout_s=6 * 3600):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        anchors = read_anchors(path)
        if all(k in anchors for k in needed):
            return anchors
        log.info(f"waiting for anchors: have {sorted(anchors)}")
        time.sleep(poll)
    raise TimeoutError(f"anchors did not complete within {timeout_s}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=int, default=5)
    ap.add_argument("--iters", type=int, default=110)
    ap.add_argument("--out", default="runs/pareto_final")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("overnight")

    anchors = wait_for_anchors(ANCHOR_LOG, ("distance-optimal", "landing-optimal"), log)
    glide_hi, phi2_loose = anchors["distance-optimal"]
    glide_lo, phi2_tight = anchors["landing-optimal"]
    log.info(f"anchors: distance-optimal glide={glide_hi:.2f} m Phi_2={phi2_loose:.2f}")
    log.info(f"         landing-optimal  glide={glide_lo:.2f} m Phi_2={phi2_tight:.2f}")

    if phi2_loose <= phi2_tight:
        log.error("Phi_2 at the distance-optimal design is not above the landing-optimal "
                  "one; there is no band to sweep. Stopping rather than guessing.")
        return 1

    # interior points only: a bound at either anchor is slack or infeasible by
    # construction, so it buys no Pareto point.
    span = phi2_loose - phi2_tight
    fracs = [0.85, 0.68, 0.50, 0.32, 0.15][:args.links]
    epsilons = [round(phi2_tight + f * span, 2) for f in fracs]
    log.info(f"band ({phi2_tight:.1f}, {phi2_loose:.1f}) -> epsilons {epsilons}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "anchors.json").write_text(json.dumps({
        "distance_optimal": {"glide_m": glide_hi, "landing_cost": phi2_loose},
        "landing_optimal": {"glide_m": glide_lo, "landing_cost": phi2_tight},
        "epsilons": epsilons,
    }, indent=2))

    cmd = [sys.executable, str(ROOT / "scripts/pareto_chain.py"),
           "--epsilons", *[str(e) for e in epsilons],
           "--iters", str(args.iters),
           "--warmup-iters", "0",          # seeded from a converged design already
           "--link-lr", "0.002", "--link-gamma", "0.995",
           "--seed-run", "distance_2D_alfix",
           "--out", args.out]
    log.info("launching chain: " + " ".join(cmd))
    with open(out / "chain.log", "w") as fh:
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    log.info(f"chain finished with code {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
