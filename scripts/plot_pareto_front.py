"""Publication figure: the epsilon-constraint Pareto front with its airfoils.

Phi_1 = glide distance (m, maximised) against Phi_2 = SoftLanding cost (minimised).
Each non-dominated design is drawn as an airfoil silhouette beside its point, so the
geometric trend along the front -- thick and cambered at the strict-landing end,
thin and efficient at the long-glide end -- is visible directly rather than only in
a table.

Writes PDF (vector, for the paper) and PNG (for slides).
"""

import argparse
import json
from pathlib import Path

import aerosandbox as asb
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/home/affinita/PhD/glider_optimization")

# Where each front point's design lives. The front is assembled from several runs,
# so the design files are keyed by (run directory, epsilon).
DESIGN_SOURCES = [
    "runs/pareto_eps_fromthick", "runs/pareto_eps8", "runs/pareto_eps9",
    "runs/tight_p1", "runs/tight_p2", "runs/tight_p3", "runs/active_a1",
]


def paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.constrained_layout.use": False,
    })


def find_design(eps):
    """The design npz for a front point, matched on its epsilon."""
    if eps is None:
        return None
    for d in DESIGN_SOURCES:
        p = ROOT / d / f"design_eps{int(eps)}.npz"
        if p.exists():
            return p
    return None


def airfoil_outline(design_npz, wandb_run=None, n=200):
    """(x, y) of the airfoil surface, normalised to unit chord."""
    if design_npz is not None:
        d = np.load(design_npz)
        upper, lower = d["upper"], d["lower"]
        le, te = float(d["le"]), float(d["te"])
    else:
        import wandb
        api = wandb.Api()
        run = sorted([r for r in api.runs("glider-optimization") if r.name == wandb_run],
                     key=lambda r: r.created_at)[-1]
        row = run.history(pandas=True).dropna(subset=["airfoil/upper_params_0"]).iloc[-1]
        upper = np.array([float(row[f"airfoil/upper_params_{i}"]) for i in range(8)])
        lower = np.array([float(row[f"airfoil/lower_params_{i}"]) for i in range(8)])
        le = float(row["airfoil/leading_edge_weight"])
        te = float(row["airfoil/TE_thickness"])

    foil = asb.KulfanAirfoil(name="a", upper_weights=upper, lower_weights=lower,
                             leading_edge_weight=le, TE_thickness=te)
    # cosine spacing: dense at the leading edge where curvature is highest
    xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yu = np.array(foil.upper_coordinates(xs)[:, 1]).ravel() if hasattr(foil, "upper_coordinates") \
        else np.array([foil.local_camber(xs) + foil.local_thickness(xs) / 2]).ravel()
    yl = np.array(foil.lower_coordinates(xs)[:, 1]).ravel() if hasattr(foil, "lower_coordinates") \
        else np.array([foil.local_camber(xs) - foil.local_thickness(xs) / 2]).ravel()
    x = np.concatenate([xs[::-1], xs])
    y = np.concatenate([yu[::-1], yl])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", default="runs/PARETO_FRONT.json")
    ap.add_argument("--out", default="results/pareto_2026-08-26/pareto_front")
    ap.add_argument("--glide-run", default="pchain_eps262",
                    help="wandb run holding the unconstrained glide-optimal design")
    args = ap.parse_args()

    paper_style()
    raw = json.loads((ROOT / args.front).read_text())
    if isinstance(raw, dict):                       # some assemblies wrap the list
        raw = raw.get("points", raw.get("front", []))
    # the front has been written with two key schemas over the course of the sweeps
    def norm_keys(p):
        return {
            "epsilon": p.get("epsilon", p.get("eps")),
            "glide_m": p.get("glide_m", p.get("glide")),
            "landing_cost": p.get("landing_cost", p.get("phi2")),
            "margin": p.get("margin"),
            "t_over_c": p.get("t_over_c", p.get("tc")),
            "camber": p.get("camber", p.get("cam")),
        }
    pts = [norm_keys(p) for p in raw]
    pts.sort(key=lambda p: p["landing_cost"])

    phi2 = np.array([p["landing_cost"] for p in pts])
    glide = np.array([p["glide_m"] for p in pts])
    tc = np.array([p["t_over_c"] for p in pts])

    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_axes([0.10, 0.13, 0.62, 0.80])

    # the front itself
    ax.plot(phi2, glide, "-", color="0.35", lw=1.2, zorder=2)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=tc.min(), vmax=tc.max())
    ax.scatter(phi2, glide, c=tc, cmap=cmap, norm=norm, s=58,
               edgecolor="black", linewidth=0.7, zorder=3)

    # region dominated by the front, shaded lightly to make "non-dominated" legible
    ax.fill_between(phi2, glide, ax.get_ylim()[0], color="0.92", zorder=1, alpha=0.6)

    ax.set_xlabel(r"landing cost  $\Phi_2$")
    ax.set_ylabel(r"glide distance  $\Phi_1$  [m]")
    ax.margins(x=0.10, y=0.14)

    # airfoil silhouettes, stacked down the right-hand margin, each linked to its point
    n = len(pts)
    for i, p in enumerate(pts):
        design = find_design(p.get("epsilon"))
        try:
            xa, ya = airfoil_outline(design, wandb_run=args.glide_run if design is None else None)
        except Exception as exc:                                   # keep the figure usable
            print(f"  (skipping airfoil for eps={p.get('epsilon')}: {exc})")
            continue

        # stack bottom-to-top in the same order as the points climb the axis, so the
        # leader lines run roughly parallel instead of crossing
        h = 0.80 / n
        y0 = 0.135 + i * h
        axf = fig.add_axes([0.755, y0 + 0.26 * h, 0.225, h * 0.52])
        colour = cmap(norm(p["t_over_c"]))
        axf.fill(xa, ya, color=colour, alpha=0.60, lw=0)
        axf.plot(xa, ya, color="black", lw=0.7)
        axf.set_xlim(-0.04, 1.04)
        axf.set_ylim(-0.20, 0.24)
        axf.set_aspect("equal")
        axf.axis("off")
        # label above its own silhouette, so it cannot be read as the neighbour's
        label = rf"$\Phi_2={p['landing_cost']:.1f}$,  $t/c={p['t_over_c']:.3f}$"
        axf.text(0.5, 1.18, label, transform=axf.transAxes, ha="center", va="bottom",
                 fontsize=7.0, color="0.15")

        # leader line from the data point to its silhouette
        con = mpl.patches.ConnectionPatch(
            xyA=(p["landing_cost"], p["glide_m"]), coordsA=ax.transData,
            xyB=(-0.06, 0.5), coordsB=axf.transAxes,
            color="0.72", lw=0.55, linestyle=(0, (2.5, 2.5)), zorder=1)
        fig.add_artist(con)

    cax = fig.add_axes([0.10, 0.965, 0.28, 0.022])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label(r"$(t/c)_{\max}$", fontsize=8, labelpad=4)
    cb.ax.tick_params(labelsize=7.5, length=2.5)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")

    # knee = point furthest from the chord joining the two extremes (standard
    # detection); argmax of the slope would just return the steepest segment's end
    x0, y0, x1, y1 = phi2[0], glide[0], phi2[-1], glide[-1]
    den = np.hypot(x1 - x0, y1 - y0)
    dist = np.abs((y1 - y0) * phi2 - (x1 - x0) * glide + x1 * y0 - y1 * x0) / den
    knee = int(np.argmax(dist))
    ax.annotate("knee", xy=(phi2[knee], glide[knee]),
                xytext=(phi2[knee] - 4.2, glide[knee] + 3.0),
                fontsize=8, color="0.35", ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color="0.65", lw=0.6))

    ax.legend(handles=[Line2D([], [], marker="o", color="0.35", markerfacecolor="0.6",
                              markeredgecolor="black", markersize=6, lw=1.2,
                              label="non-dominated designs")],
              loc="upper left")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"  wrote {out}.pdf and {out}.png  ({len(pts)} points)")


if __name__ == "__main__":
    main()
