"""Publication figure per task: trajectories, design change, shape evolution, convergence.

Four panels side by side for one task:

  (a) flight paths at the seed design and at the optimised design, one line per
      launch condition, so the manoeuvre and what the optimiser bought are visible
      together. Any trajectory that climbs back up after already having descended
      is flagged (dotted + a warning marker) rather than plotted as if it were a
      normal result -- unpowered flight has no business re-gaining altitude this
      way, and it has shown up before in re-solves of an unconverged design.
  (b) seed vs. optimised airfoil, superimposed on one axis -- just the two shapes
      being compared, not the whole iteration history (overlaying nine
      near-identical intermediate outlines made it impossible to see what actually
      changed).
  (c) the shape parameters that (b) can't show a trend for: max thickness and max
      camber over the outer iterations.
  (d) the objective against iteration.

Trajectories are read from the traj_<task>_<launch>_<iteration>.npy files the runner
writes; make_task_trajectories.py regenerates them for a frozen design.
"""

import argparse
import re
from pathlib import Path

import aerosandbox as asb
import matplotlib as mpl
import numpy as np
import wandb
from matplotlib import pyplot as plt

ROOT = Path("/home/affinita/PhD/glider_optimization")

# check_monotonic_climb: flag a trajectory that re-climbs after already descending.
# Only meaningful for glide -- the objective is a smooth, efficient trade of altitude
# for range, so a resurgent climb is a red flag. Perching and landing *end* with a
# deliberate flare/arrest manoeuvre (nose up, oscillating settle to touchdown), so the
# same check would flag every normal result there and just be noise.
TASKS = {
    "landing":  {"run": "best_landing",  "traj": "softlanding",
                 "label": "Landing",  "obj": r"$\Phi_{\mathrm{land}}$", "check_monotonic_climb": False},
    "perching": {"run": "best_perching", "traj": "perching",
                 "label": "Perching", "obj": r"$\Phi_{\mathrm{perch}}$", "check_monotonic_climb": False},
    "glide":    {"run": "best_glide",    "traj": "distance",
                 "label": "Gliding",  "obj": "glide distance [m]", "check_monotonic_climb": True},
    "perching_3d": {"run": "perching3D", "traj": "perching",
                 "label": "Perching (3D wing, root section)", "obj": r"$\Phi_{\mathrm{perch}}$",
                 "check_monotonic_climb": False},
}


def paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def history(run_name):
    api = wandb.Api()
    run = sorted([r for r in api.runs("glider-optimization") if r.name == run_name],
                 key=lambda r: r.created_at)[-1]
    return run.history(pandas=True)


def kulfan_row(row):
    return dict(upper=[row[f"airfoil/upper_params_{k}"] for k in range(8)],
                lower=[row[f"airfoil/lower_params_{k}"] for k in range(8)],
                le=row["airfoil/leading_edge_weight"], te=row["airfoil/TE_thickness"])


def airfoil_xy(upper, lower, le, te, n=180):
    foil = asb.KulfanAirfoil(name="a", upper_weights=np.asarray(upper),
                             lower_weights=np.asarray(lower),
                             leading_edge_weight=float(le), TE_thickness=float(te))
    xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))       # dense at the leading edge
    cam, thk = foil.local_camber(xs), foil.local_thickness(xs)
    yu, yl = np.asarray(cam + thk / 2).ravel(), np.asarray(cam - thk / 2).ravel()
    return np.concatenate([xs[::-1], xs]), np.concatenate([yu[::-1], yl]), np.asarray(cam), np.asarray(thk)


def load_trajectories(task, which):
    """{launch: state array} from results/figures/traj_<task>_<which>/."""
    d = ROOT / f"results/figures/traj_{task}_{which}"
    if not d.is_dir():
        print(f"  WARNING: no {d} -- panel (a) will be missing the {which} trajectories "
              f"(run make_task_trajectories.py {task} first)")
        return {}
    pat = re.compile(r"traj_\w+?_(\d+)_(\d+)\.npy$")
    out = {}
    for p in sorted(d.glob("*.npy")):
        m = pat.search(p.name)
        if m:
            out[int(m.group(1))] = np.load(p)
    if not out:
        print(f"  WARNING: {d} exists but has no trajectory files")
    return dict(sorted(out.items()))


OBJ_COL = "evaluation/objective_total"


def best_row(h):
    """The logged row with the lowest raw objective (glide's is a negated distance,
    so "lowest" is uniformly "best"), not the literally-last one -- once the LR has
    decayed and the objective has plateaued, Adam just noisily wanders +/-2% around
    it rather than monotonically improving, so a run stopped early (or just one whose
    schedule has decayed a lot by the end) doesn't have its best design in the last
    logged row."""
    if OBJ_COL in h.columns:
        sub = h.dropna(subset=[OBJ_COL])
        if not sub.empty:
            return sub.loc[sub[OBJ_COL].idxmin()]
    return h.iloc[-1]


def flag_nonmonotonic_climb(z, rise_thresh=0.35, min_drop=0.4):
    """True if z(t) re-climbs by more than `rise_thresh` after already having
    dropped more than `min_drop` from its start -- unpowered flight can only
    trade speed for altitude, not manufacture a second climb from nothing, so
    this is a good proxy for a spurious (non-converged-design, cold-solve)
    local optimum rather than a real manoeuvre. A single climb-then-descend
    arc (e.g. perching's flare) never triggers it, since the running minimum
    never drops before the climb happens."""
    z0 = z[0]
    zmin = z0
    for i in range(1, len(z)):
        if z[i] < zmin:
            zmin = z[i]
        elif z[i] - zmin > rise_thresh and (z0 - zmin) > min_drop:
            return True, i
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = TASKS[args.task]
    paper_style()
    h = history(cfg["run"]).dropna(subset=["airfoil/upper_params_0"])

    fig = plt.figure(figsize=(9.6, 2.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.85, 0.95, 0.95], wspace=0.48,
                          left=0.06, right=0.99, bottom=0.34, top=0.87)
    ax_tr, ax_af, ax_eq, ax_cv = (fig.add_subplot(gs[0, i]) for i in range(4))

    # ---- (a) trajectories ----------------------------------------------------
    trajs = load_trajectories(args.task, "opt")
    seed_trajs = load_trajectories(args.task, "seed")

    for k, st in seed_trajs.items():
        ax_tr.plot(st[:, 0], st[:, 1], color="0.72", lw=0.9, zorder=1,
                   label="seed design" if k == 0 else None)
    x_varies = len({round(float(st[0, 0]), 2) for st in trajs.values()}) > 1
    cols = plt.get_cmap("plasma")(np.linspace(0.12, 0.82, max(len(trajs), 1)))
    any_flagged = False
    for (k, st), c in zip(trajs.items(), cols):
        lbl = (rf"$(x_0,z_0)=({st[0, 0]:.0f},{st[0, 1]:.0f})$"
               if x_varies else rf"$z_0={st[0, 1]:.1f}$ m")
        flagged, i_flag = (flag_nonmonotonic_climb(st[:, 1]) if cfg["check_monotonic_climb"]
                           else (False, None))
        if flagged:
            any_flagged = True
            lbl += " (*)"
            print(f"  WARNING: {args.task} trajectory {k} (z0={st[0,1]:.1f}) re-climbs "
                  f"{st[i_flag, 1] - st[:i_flag, 1].min():.2f} m after descending "
                  f"-- likely a spurious cold-solve local optimum, not a real manoeuvre")
        ax_tr.plot(st[:, 0], st[:, 1], color=c, lw=1.4, zorder=3, label=lbl,
                   ls=(":" if flagged else "-"))
        ax_tr.plot(st[0, 0], st[0, 1], "o", color=c, ms=3.2, zorder=4)
    ax_tr.axhline(0.0, color="0.25", lw=0.8, zorder=2)
    ax_tr.set_xlabel(r"$x$ [m]"); ax_tr.set_ylabel(r"$z$ [m]")
    title = f"({chr(97)}) {cfg['label']}: flight paths"
    if any_flagged:
        title += " (* = suspect)"
    ax_tr.set_title(title, loc="left", pad=4)
    ax_tr.legend(loc="best", handlelength=1.4, fontsize=6.6)

    # ---- (b) seed vs. optimised section, superimposed -------------------------
    seed_row, opt_row = h.iloc[0], best_row(h)
    xs_seed, ys_seed, _, _ = airfoil_xy(**kulfan_row(seed_row))
    xs_opt, ys_opt, _, _ = airfoil_xy(**kulfan_row(opt_row))
    ax_af.plot(xs_seed, ys_seed, color="0.55", lw=1.4, ls="--", label="seed", zorder=2)
    ax_af.fill(xs_seed, ys_seed, color="0.55", alpha=0.12, zorder=1)
    ax_af.plot(xs_opt, ys_opt, color="#c98a12", lw=1.8, label="optimised", zorder=4)
    ax_af.fill(xs_opt, ys_opt, color="#c98a12", alpha=0.22, zorder=3)
    ax_af.set_aspect("equal"); ax_af.set_xlim(-0.03, 1.03)
    ax_af.set_xlabel(r"$x/c$")
    ax_af.set_title("(b) seed vs. optimised", loc="left", pad=4)
    ax_af.legend(loc="upper right", handlelength=1.4)

    # ---- (c) shape parameters vs. iteration -----------------------------------
    tcs, cams = [], []
    for _, row in h.iterrows():
        _, _, cam, thk = airfoil_xy(**kulfan_row(row))
        tcs.append(float(thk.max())); cams.append(float(np.abs(cam).max()))
    steps = h["_step"].to_numpy()
    tcs, cams = np.array(tcs), np.array(cams)
    # Plotted as a ratio to the seed value on one shared axis rather than a twin-axis
    # (t/c, camber): a second y-axis for camber needs its own margin on panel (c)'s
    # right side, which collides with panel (d)'s y-label sitting right next to it.
    # Normalising both to "x1.0 at the seed" also makes their rates of change directly
    # comparable, which two independently-scaled axes wouldn't.
    ax_eq.axhline(1.0, color="0.85", lw=0.8, zorder=0)
    ax_eq.plot(steps, tcs / tcs[0], color="#1b3a6b", lw=1.4, label="max t/c")
    ax_eq.plot(steps, cams / cams[0], color="#c9542c", lw=1.4, label="max camber")
    ax_eq.set_xlabel("iteration"); ax_eq.set_ylabel(r"$\times$ seed value")
    ax_eq.set_title("(c) shape vs. iteration", loc="left", pad=4)
    ax_eq.legend(loc="best", fontsize=6.6, handlelength=1.4)

    # ---- (d) convergence --------------------------------------------------------
    key = OBJ_COL if OBJ_COL in h.columns else next(
        (c for c in ("Cost", "evaluation/cost", "Objective") if c in h.columns), None)
    if key is None:
        num = [c for c in h.columns if h[c].dtype.kind == "f" and "airfoil" not in c]
        key = num[0] if num else None
    if key:
        series = h[["_step", key]].dropna()
        y = series[key].to_numpy()
        if args.task == "glide":
            y = -y                                   # report distance, not its negation
        ax_cv.plot(series["_step"], y, color="0.2", lw=1.2)
        best_step = int(opt_row["_step"])
        best_mask = series["_step"] == best_step
        ax_cv.scatter(series.loc[best_mask, "_step"], y[best_mask.to_numpy()],
                      s=26, color="#c98a12", zorder=3, label="design used in (a)/(b)")
        ax_cv.legend(loc="best", fontsize=6.2, handlelength=1.2)
    ax_cv.set_xlabel("iteration"); ax_cv.set_ylabel(cfg["obj"])
    ax_cv.set_title("(d) convergence", loc="left", pad=4)

    out = Path(args.out) if args.out else ROOT / f"results/figures/{args.task}_summary"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"  wrote {out}.pdf / .png   ({len(trajs)} trajectories, {len(h)} design iterations)")


if __name__ == "__main__":
    main()
