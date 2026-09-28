"""Compare the airfoils that different objectives select.

Pulls the final Kulfan parameters of each wandb run and reports the geometry that
the optimiser actually chose -- thickness, camber, and the aerodynamics that
follow -- so a claim like "the landing-optimal wing should be thicker" can be
checked rather than assumed.

Usage:
    python scripts/compare_airfoils.py distance_2D_alfix landing_only_2D ...
    python scripts/compare_airfoils.py --project glider-optimization run_a run_b
"""

from __future__ import annotations

import argparse

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare final airfoils across runs")
    p.add_argument("runs", nargs="+", help="wandb run names")
    p.add_argument("--project", default="glider-optimization")
    p.add_argument("--entity", default=None)
    p.add_argument("--seed-config", default="conf/pareto_2D.yaml",
                   help="config whose initial airfoil is used as the 'seed' reference row")
    return p.parse_args(argv)


def kulfan_geometry(upper, lower, LE, TE):
    """Max thickness, its chordwise location, and max camber, from Kulfan weights."""
    import aerosandbox as asb

    foil = asb.KulfanAirfoil(
        name="x",
        upper_weights=np.asarray(upper),
        lower_weights=np.asarray(lower),
        leading_edge_weight=float(LE),
        TE_thickness=float(TE),
    )
    x = np.linspace(0.0, 1.0, 400)
    thickness = foil.local_thickness(x)
    camber = foil.local_camber(x)
    return {
        "t/c max": float(np.max(thickness)),
        "x(t max)": float(x[int(np.argmax(thickness))]),
        "camber max": float(np.max(np.abs(camber))),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    import wandb
    from pathlib import Path
    from glider_optimization.config import load_config

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project

    rows = []
    seed = load_config(Path(args.seed_config)).airfoil
    rows.append(("seed (initial)", kulfan_geometry(
        seed.upper_initial_weights, seed.lower_initial_weights,
        seed.leading_edge_weight, seed.TE_thickness)))

    for name in args.runs:
        matches = [r for r in api.runs(path) if r.name == name]
        if not matches:
            print(f"  ! no wandb run named {name}")
            continue
        run = sorted(matches, key=lambda r: r.created_at)[-1]
        hist = run.history(pandas=True)
        have = hist.dropna(subset=["airfoil/upper_params_0"])
        if have.empty:
            print(f"  ! {name} logged no airfoil parameters")
            continue
        row = have.iloc[-1]
        rows.append((name, kulfan_geometry(
            [row[f"airfoil/upper_params_{i}"] for i in range(8)],
            [row[f"airfoil/lower_params_{i}"] for i in range(8)],
            row["airfoil/leading_edge_weight"], row["airfoil/TE_thickness"])))

    width = max(len(n) for n, _ in rows)
    keys = list(rows[0][1])
    print(f"{'design':<{width}} | " + " | ".join(f"{k:>10}" for k in keys))
    print("-" * (width + 3 + 13 * len(keys)))
    for name, geom in rows:
        print(f"{name:<{width}} | " + " | ".join(f"{geom[k]:>10.4f}" for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
