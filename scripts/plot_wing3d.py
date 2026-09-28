"""Stunning 3D wing visualisation for the spanwise (lifting-line) co-design task.

Lofts the optimised root/tip Kulfan sections into a full-span wing skin and
colours it by the sectional lift coefficient from a real LLT solve at a
reference flight condition -- the same physics the design was optimised
against, not a cosmetic gradient. Produces a static hero PNG/PDF and an
interactive HTML (rotate/zoom/hover) via plotly.
"""

import argparse
from pathlib import Path

import aerosandbox as asb
import matplotlib as mpl
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from glider_optimization.config import load_config
from glider_optimization.utils import llt
from glider_optimization.utils.spanwise_geometry import build_half_wing_stations_from_cfg

ROOT = Path("/home/affinita/PhD/glider_optimization")
N_STATIONS_HALF = 41   # half-span stations -> 2*N-1 panels after mirroring
N_CHORD = 100           # points per airfoil outline


def latest_run(run_name: str):
    api = wandb.Api()
    runs = sorted([r for r in api.runs("glider-optimization") if r.name == run_name],
                  key=lambda r: r.created_at)
    if not runs:
        raise SystemExit(f"no wandb run named {run_name!r}")
    return runs[-1]


def design_from_wandb(run_name: str):
    h2 = latest_run(run_name).history(pandas=True).dropna(subset=["airfoil/upper_params_0"])
    # Best-by-objective, not literally-last -- see collect_experiment.py's best_row.
    if "evaluation/objective_total" in h2.columns:
        sub = h2.dropna(subset=["evaluation/objective_total"])
        row = sub.loc[sub["evaluation/objective_total"].idxmin()] if not sub.empty else h2.iloc[-1]
    else:
        row = h2.iloc[-1]
    def vec(prefix, n=8):
        return np.array([float(row[f"{prefix}_{i}"]) for i in range(n)])
    return {
        "upper": vec("airfoil/upper_params"), "lower": vec("airfoil/lower_params"),
        "le": float(row["airfoil/leading_edge_weight"]), "te": float(row["airfoil/TE_thickness"]),
        "upper_tip": vec("airfoil/upper_params_tip"), "lower_tip": vec("airfoil/lower_params_tip"),
        "le_tip": float(row["airfoil/leading_edge_weight_tip"]), "te_tip": float(row["airfoil/TE_thickness_tip"]),
        "step": int(row["_step"]),
    }


def airfoil_outline(upper, lower, le, te, n=N_CHORD):
    """Closed (x/c, y/c) outline: upper surface LE->TE then lower TE->LE."""
    foil = asb.KulfanAirfoil(name="a", upper_weights=upper, lower_weights=lower,
                             leading_edge_weight=le, TE_thickness=te)
    xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    cam, thk = np.asarray(foil.local_camber(xs)).ravel(), np.asarray(foil.local_thickness(xs)).ravel()
    yu, yl = cam + thk / 2, cam - thk / 2
    x = np.concatenate([xs[::-1], xs])
    y = np.concatenate([yu[::-1], yl])
    return x, y


def build_loft(design, wing_cfg, n_stations=N_STATIONS_HALF, chord_exaggeration=1.0):
    """Full-span (x, y, z) skin mesh, shape (2*n_stations-1, 2*N_CHORD).

    Real sailplane wings at this aspect ratio (span/chord ~ 40) render as an
    invisible sliver at true scale, so `chord_exaggeration` inflates the
    chordwise/thickness extent about each station's leading edge while leaving
    the spanwise geometry (and hence sweep/taper) physically correct -- purely
    a rendering aid, always labelled on the plot.
    """
    half = build_half_wing_stations_from_cfg(wing_cfg, n_span_stations=n_stations)
    y_full, c_full, xle_full, _ = llt.mirror_full(
        half["y_half"], half["c_half"], half["xle_half"], half["twist_half"])
    eta_full = np.clip(np.abs(y_full) / max(half["y_half"][-1], 1e-9), 0.0, 1.0)

    X = np.zeros((len(y_full), 2 * N_CHORD))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for i, (y, c, xle, eta) in enumerate(zip(y_full, c_full, xle_full, eta_full)):
        up = (1 - eta) * design["upper"] + eta * design["upper_tip"]
        lo = (1 - eta) * design["lower"] + eta * design["lower_tip"]
        le = (1 - eta) * design["le"] + eta * design["le_tip"]
        te = (1 - eta) * design["te"] + eta * design["te_tip"]
        xc, yc = airfoil_outline(up, lo, le, te)
        X[i] = xle + xc * c * chord_exaggeration
        Z[i] = yc * c * chord_exaggeration
        Y[i] = y
    return X, Y, Z, y_full, half


def section_cl(design, wing_cfg, alpha_deg, V, n_stations=N_STATIONS_HALF, device="cpu", model_size="xxxlarge"):
    """Real per-panel sectional CL from an LLT fixed-point solve, root panel -> tip panel."""
    half = build_half_wing_stations_from_cfg(wing_cfg, n_span_stations=n_stations)
    comp = llt.build_llt_system(half["y_half"], half["c_half"], half["xle_half"], half["twist_half"],
                                 dihedral_deg=half["dihedral_deg"])

    def T(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device)

    eta = np.clip(np.abs(comp["y_mid"]) / max(half["y_half"][-1], 1e-9), 0.0, 1.0)
    eta_t = T(eta)
    upper = (1 - eta_t)[:, None] * T(design["upper"]) + eta_t[:, None] * T(design["upper_tip"])
    lower = (1 - eta_t)[:, None] * T(design["lower"]) + eta_t[:, None] * T(design["lower_tip"])
    le = (1 - eta_t) * T([design["le"]]) + eta_t * T([design["le_tip"]])
    te = (1 - eta_t) * T([design["te"]]) + eta_t * T([design["te_tip"]])

    alpha_t = T([alpha_deg]).reshape(1, 1)
    V_t = T([V]).reshape(1, 1)
    C, alpha_eff_pan, Re_pan = llt.LLTImplicitFn.apply(
        alpha_t, V_t, upper, lower, le, te,
        T(comp["dy"]), T(comp["c_mid"]), T(comp["tw_mid"]), T(comp["S"]), T(comp["cbar"]),
        T(comp["D_nf"]), T(comp["D_tr"]), T(comp["mirror_of"], dtype=torch.long),
        T(comp["cos_sweep"]), T(comp["cos2_sweep"]), T(1.225), T(1.789e-5),
        0.4, 1e-3, 200, True, model_size,
    )
    cl_pan = llt.eval_nf_batched(upper, lower, le, te, alpha_eff_pan, Re_pan, model_size)["CL"][0]
    return comp["y_mid"], cl_pan.detach().cpu().numpy(), float(C[0, 0]), float(C[0, 1])


def paper_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif", "font.size": 10,
        "axes.labelsize": 10.5, "axes.titlesize": 11,
    })


CHORD_EXAGGERATION = 9.0


ROOT_COLOR = "#1b3a6b"   # deep navy -- root section
TIP_COLOR = "#e0752a"    # warm orange -- tip section


def plot_static(X, Y, Z, y_mid, cl_mid, CL, CD, alpha_deg, V, out, label, design, chord_root, chord_tip):
    cl_full = np.interp(Y[:, 0], y_mid, cl_mid)
    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=cl_mid.min(), vmax=cl_mid.max())
    facecolors = cmap(norm(cl_full))[:, None, :].repeat(X.shape[1], axis=1)
    # y_mid runs root(-tip) -> root -> tip across the full mirrored span, so the
    # *root* section is the middle panel (y ~ 0) and either end is a wingtip --
    # indexing the two ends as "root" and "tip" (as if root were at y_mid[0])
    # silently compares two wingtips, which are near-identical by symmetry.
    root_i = int(np.argmin(np.abs(y_mid)))
    tip_i = int(np.argmax(np.abs(y_mid)))
    cl_root, cl_tip = float(cl_mid[root_i]), float(cl_mid[tip_i])

    paper_style()
    fig = plt.figure(figsize=(10, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.12, hspace=0.42, left=0.02, right=0.985, top=0.84, bottom=0.14)

    # ---- (a) 3D loft, hero panel --------------------------------------------
    # A fixed rect (not queried post-hoc) so the horizontal colorbar below can be
    # placed reliably -- Axes3D.get_position() reports the whole square subplot
    # cell, not the actual rendered footprint, which silently drifts with `zoom`.
    rect = [0.015, 0.10, 0.48, 0.74]
    ax = fig.add_axes(rect, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=2,
                     linewidth=0, antialiased=True, shade=True)
    span = np.ptp(Y)
    ax.set_box_aspect((np.ptp(X), span, max(np.ptp(Z), 0.08 * span)), zoom=1.85)
    ax.set_xticks([]); ax.set_zticks([])
    ax.set_ylabel("span [m]")
    ax.xaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
    ax.view_init(elev=18, azim=-72)
    fig.text(0.015, 0.965, f"({chr(97)}) {label}: optimised wing, coloured by sectional $C_l$",
              fontsize=9.5, ha="left", va="top")
    fig.text(0.015, 0.925,
              rf"(chord/thickness exaggerated {CHORD_EXAGGERATION:.0f}$\times$ for visibility)"
              rf"$\quad\alpha={alpha_deg:.1f}^\circ,\ V={V:.1f}$ m/s $\to C_L={CL:.3f}$, $C_D={CD:.4f}$",
              fontsize=8.5, ha="left", va="top", color="0.25")

    # ---- (b) root vs tip section, superimposed at true relative scale -------
    # Both sections are LE-aligned at the origin; the tip is drawn at chord_tip/
    # chord_root of the root's size (not renormalised to its own unit chord), so
    # the overlay also shows the taper, not just the two shapes.
    ax_af = fig.add_subplot(gs[0, 1])
    xr, yr = airfoil_outline(design["upper"], design["lower"], design["le"], design["te"])
    xt, yt = airfoil_outline(design["upper_tip"], design["lower_tip"], design["le_tip"], design["te_tip"])
    tc = chord_tip / chord_root
    ax_af.plot(xr, yr, color=ROOT_COLOR, lw=2.0, label=rf"root ($C_l$={cl_root:.2f})", zorder=3)
    ax_af.fill(xr, yr, color=ROOT_COLOR, alpha=0.18, zorder=1)
    ax_af.plot(xt * tc, yt * tc, color=TIP_COLOR, lw=2.0, ls="--", label=rf"tip ($C_l$={cl_tip:.2f})", zorder=4)
    ax_af.fill(xt * tc, yt * tc, color=TIP_COLOR, alpha=0.28, zorder=2)
    ax_af.set_aspect("equal"); ax_af.set_xlim(-0.03, 1.03)
    ax_af.set_xlabel(r"$x/c_{\mathrm{root}}$")
    ax_af.set_title("(b) root vs. tip section, superimposed (true relative scale)", loc="left", pad=4, fontsize=9.2)
    ax_af.legend(loc="upper right", handlelength=1.8, fontsize=7.5)

    # ---- (c) spanwise Cl distribution ---------------------------------------
    ax_cl = fig.add_subplot(gs[1, 1])
    ax_cl.plot(y_mid, cl_mid, color="0.15", lw=1.6)
    sc = ax_cl.scatter(y_mid, cl_mid, c=cl_mid, cmap=cmap, norm=norm, s=14, zorder=3)
    ax_cl.scatter([y_mid[root_i]], [cl_root], s=70, facecolors="none",
                  edgecolors=ROOT_COLOR, linewidths=1.8, zorder=4)
    ax_cl.scatter([y_mid[tip_i]], [cl_tip], s=70, facecolors="none",
                  edgecolors=TIP_COLOR, linewidths=1.8, zorder=4)
    ax_cl.axvline(0.0, color="0.75", lw=0.8, zorder=0)
    ax_cl.set_xlabel(r"span station $y$ [m]"); ax_cl.set_ylabel(r"sectional $C_l$")
    ax_cl.set_title("(c) spanwise lift distribution", loc="left", pad=4)
    cbar = fig.colorbar(sc, ax=ax_cl, pad=0.02, aspect=14)
    cbar.set_label("$C_l$  (also (a)'s colour scale)", fontsize=7.5, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"  wrote {out}.png / .pdf")


def plot_interactive(X, Y, Z, y_mid, cl_mid, CL, CD, alpha_deg, V, out, label):
    import plotly.graph_objects as go
    cl_full = np.interp(Y[:, 0], y_mid, cl_mid)
    surfacecolor = np.repeat(cl_full[:, None], X.shape[1], axis=1)
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z, surfacecolor=surfacecolor, colorscale="Plasma",
        colorbar=dict(title="sectional C_l"), lighting=dict(ambient=0.55, diffuse=0.7, specular=0.35),
        hovertemplate="span %{y:.3f} m<br>C_l %{surfacecolor:.3f}<extra></extra>",
    )])
    fig.update_layout(
        title=f"{label}: optimised 3D wing (LLT), chord exaggerated {CHORD_EXAGGERATION:.0f}x for visibility"
              f"<br>alpha={alpha_deg:.1f} deg, V={V:.1f} m/s -> CL={CL:.3f}, CD={CD:.4f}",
        scene=dict(aspectmode="data",
                   xaxis_title="chord [m] (exaggerated)", yaxis_title="span [m]", zaxis_title="",
                   xaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(f"{out}.html", include_plotlyjs="cdn")
    print(f"  wrote {out}.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="perching3D")
    ap.add_argument("--conf", default="conf/perching_3D.yaml")
    ap.add_argument("--label", default="Perching (3D wing)")
    ap.add_argument("--alpha", type=float, default=6.0, help="reference AoA [deg]")
    ap.add_argument("--V", type=float, default=6.0, help="reference airspeed [m/s]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.conf)
    wing_cfg = cfg.plane.get("wing", {})
    design = design_from_wandb(args.run)

    X, Y, Z, y_full, half = build_loft(design, wing_cfg, chord_exaggeration=CHORD_EXAGGERATION)
    y_mid, cl_mid, CL, CD = section_cl(design, wing_cfg, args.alpha, args.V)
    chord_root, chord_tip = float(half["c_half"][0]), float(half["c_half"][-1])

    out = Path(args.out) if args.out else ROOT / "results/figures/wing3d_perching"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_static(X, Y, Z, y_mid, cl_mid, CL, CD, args.alpha, args.V, out, args.label, design,
                chord_root, chord_tip)
    plot_interactive(X, Y, Z, y_mid, cl_mid, CL, CD, args.alpha, args.V, out, args.label)
    print(f"  design from wandb run {args.run!r} @ step {design['step']}")


if __name__ == "__main__":
    main()
