"""Cross-task hero comparison: one designer, four missions, four different wings.

Reads experiments/<task>/design.json (written by collect_experiment.py) for every
task that has one yet, and produces a single "the shape follows the mission"
figure: optimised sections overlaid on a shared chord, plus the geometric
trade-off (thickness vs. camber) that separates them.
"""
import argparse
import json
from pathlib import Path

import aerosandbox as asb
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

ROOT = Path("/home/affinita/PhD/glider_optimization")

TASKS = [
    ("glide",       "Max distance",  "#3b7dd8"),
    ("perching",    "Perching",      "#d9534f"),
    ("landing",     "Soft landing",  "#e8a33d"),
    ("perching_3d", "Perching (3D)", "#5fa55a"),
]


def paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif", "font.size": 10,
        "axes.labelsize": 10.5, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def outline(design, root=True, n=180):
    suf = "" if root else "_tip"
    foil = asb.KulfanAirfoil(
        name="a", upper_weights=np.asarray(design[f"upper{suf}"]), lower_weights=np.asarray(design[f"lower{suf}"]),
        leading_edge_weight=design[f"leading_edge_weight{suf}"], TE_thickness=design[f"TE_thickness{suf}"])
    xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    cam, thk = np.asarray(foil.local_camber(xs)).ravel(), np.asarray(foil.local_thickness(xs)).ravel()
    return foil, xs, cam, thk


def tc_camber(design, root=True):
    _, xs, cam, thk = outline(design, root=root)
    return float(thk.max()), float(np.abs(cam).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/overview"))
    args = ap.parse_args()

    entries = []
    for key, label, color in TASKS:
        p = ROOT / f"experiments/{key}/design.json"
        if not p.exists():
            print(f"  skip {key}: no {p}")
            continue
        info = json.loads(p.read_text())
        entries.append((key, label, color, info))

    if not entries:
        raise SystemExit("no experiments/<task>/design.json found -- run collect_experiment.py first")

    paper_style()
    fig, (ax_af, ax_tc) = plt.subplots(1, 2, figsize=(9.5, 3.6), width_ratios=[1.4, 1.0])

    tc_vals, cam_vals, labels, colors = [], [], [], []
    for key, label, color, info in entries:
        design = info["design"]
        _, xs, cam, thk = outline(design, root=True)
        yu, yl = cam + thk / 2, cam - thk / 2
        x = np.concatenate([xs[::-1], xs]); y = np.concatenate([yu[::-1], yl])
        ax_af.plot(x, y, color=color, lw=1.8, label=label)
        ax_af.fill(x, y, color=color, alpha=0.12)
        tc, cm_ = tc_camber(design, root=True)
        tc_vals.append(tc * 100); cam_vals.append(cm_ * 100)
        labels.append(label); colors.append(color)

    ax_af.set_aspect("equal"); ax_af.set_xlim(-0.03, 1.03)
    ax_af.axhline(0, color="0.85", lw=0.7, zorder=0)
    ax_af.set_xlabel(r"$x/c$"); ax_af.set_title("(a) optimised sections, shared chord", loc="left")
    ax_af.legend(loc="upper right", handlelength=1.6, fontsize=8)

    x = np.arange(len(labels))
    w = 0.36
    ax_tc.bar(x - w / 2, tc_vals, width=w, color=colors, alpha=0.9, label="max t/c")
    ax_tc.bar(x + w / 2, cam_vals, width=w, color=colors, alpha=0.45, hatch="//", label="max camber")
    ax_tc.set_xticks(x); ax_tc.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax_tc.set_ylabel("% chord")
    ax_tc.set_title("(b) thickness vs. camber", loc="left")
    ax_tc.legend(loc="upper right", fontsize=8)

    fig.suptitle("Co-designed wing sections: the optimum follows the mission", y=1.02, fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"  wrote {out}.png / .pdf  ({len(entries)} tasks: {[e[0] for e in entries]})")


if __name__ == "__main__":
    main()
