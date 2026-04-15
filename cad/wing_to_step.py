#!/usr/bin/env python3
"""
wing_to_step.py  —  Convert wing geometry to a STEP CAD file.

Accepted inputs:
  .xml   flow5 wing export  (NACA / named foils, 2+ sections)
  .yaml  flexible definition (see wing_cad.yaml for an example):
           - Named / NACA foils via AeroSandbox  →  airfoil_root / airfoil_tip
           - CST / Kulfan foils                  →  kulfan_root  / kulfan_tip
         When root ≠ tip all Kulfan weights are linearly interpolated along
         the span at every station listed in y_half.  No extra discretisation
         stations are required; the y_half array already defines the loft
         skeleton.

Usage
-----
    conda run -n general python wing_to_step.py Main_wing.xml
    conda run -n general python wing_to_step.py wing_cad.yaml
    conda run -n general python wing_to_step.py wing_cad.yaml --out optimised.step
    conda run -n general python wing_to_step.py wing_cad.yaml --half
    conda run -n general python wing_to_step.py wing_cad.yaml --n-pts 256

Dependencies (conda-forge): cadquery  aerosandbox  pyyaml  numpy
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
import aerosandbox as asb
import cadquery as cq


# ---------------------------------------------------------------------------
# Airfoil helpers
# ---------------------------------------------------------------------------

def _kulfan_from_dict(d: dict) -> asb.KulfanAirfoil:
    """Build a KulfanAirfoil from a YAML kulfan_* sub-dict."""
    return asb.KulfanAirfoil(
        upper_weights=np.array(d["upper_weights"], dtype=float),
        lower_weights=np.array(d["lower_weights"], dtype=float),
        leading_edge_weight=float(d.get("leading_edge_weight", 0.0)),
        TE_thickness=float(d.get("TE_thickness", 0.0)),
    )


def _lerp_kulfan(root: asb.KulfanAirfoil,
                 tip:  asb.KulfanAirfoil,
                 eta:  float) -> asb.KulfanAirfoil:
    """
    Linearly interpolate every Kulfan parameter from root (eta=0) to tip (eta=1).
    This is exact in the CST sense: the interpolated shape is algebraically the
    weighted sum of the two CST curves.
    """
    w = 1.0 - eta
    return asb.KulfanAirfoil(
        upper_weights       = w * np.array(root.upper_weights)  + eta * np.array(tip.upper_weights),
        lower_weights       = w * np.array(root.lower_weights)  + eta * np.array(tip.lower_weights),
        leading_edge_weight = w * root.leading_edge_weight       + eta * tip.leading_edge_weight,
        TE_thickness        = w * root.TE_thickness              + eta * tip.TE_thickness,
    )


def _normalize_foil_name(name: str) -> str:
    """Normalize airfoil name to a form aerosandbox accepts.  'NACA 4412' → 'naca4412'."""
    return name.lower().replace(" ", "")


def _normalized_coords(foil: asb.KulfanAirfoil | str, n_pts: int):
    """
    Return (upper, lower) normalized coordinate arrays, each shape (n_pts, 2),
    both running LE (x=0) → TE (x=1).

    foil  – an asb.KulfanAirfoil  OR  a name string accepted by asb.Airfoil()
    """
    af = foil.repanel(n_points_per_side=n_pts) if isinstance(foil, asb.KulfanAirfoil) \
         else asb.Airfoil(_normalize_foil_name(foil)).repanel(n_points_per_side=n_pts)

    c  = af.coordinates                    # (2·n_pts-1, 2),  TE → LE → TE
    le = int(np.argmin(c[:, 0]))           # leading-edge index  (x ≈ 0)
    upper = c[le::-1]                      # LE → TE  upper surface
    lower = c[le:]                         # LE → TE  lower surface
    return upper, lower


# ---------------------------------------------------------------------------
# 3-D profile wire
# ---------------------------------------------------------------------------

def _make_wire(foil, chord: float, x_off: float,
               y_span: float, z_dih: float,
               twist_deg: float, n_pts: int) -> cq.Wire:
    """
    Build a closed CadQuery spline Wire for one wing cross-section.

    Coordinate convention (AVL / flow5 / standard):
        X  chordwise, positive toward trailing edge
        Y  spanwise,  positive toward tip
        Z  thickness, positive upward
    """
    upper_n, lower_n = _normalized_coords(foil, n_pts)

    def to_3d(pts: np.ndarray) -> list:
        if twist_deg != 0.0:
            a   = np.radians(twist_deg)
            rot = np.array([[np.cos(a), -np.sin(a)],
                            [np.sin(a),  np.cos(a)]])
            pts = pts @ rot.T
        return [cq.Vector(x_off + p[0] * chord, y_span, z_dih + p[1] * chord)
                for p in pts]

    upper_edge = cq.Edge.makeSpline(to_3d(upper_n))
    lower_edge = cq.Edge.makeSpline(to_3d(lower_n[::-1]))   # TE → LE to close
    return cq.Wire.assembleEdges([upper_edge, lower_edge])


# ---------------------------------------------------------------------------
# YAML input
# ---------------------------------------------------------------------------

def _foil_at_eta(wing: dict, eta: float) -> asb.KulfanAirfoil | str:
    """
    Return the airfoil definition interpolated at spanwise fraction eta ∈ [0,1].

    Priority:
      1. kulfan_root + kulfan_tip  →  CST linear interpolation
      2. airfoil_root + airfoil_tip (different) →  convert to Kulfan, then lerp
      3. airfoil_root == airfoil_tip (or only one given)  →  single named foil
    """
    if "kulfan_root" in wing and "kulfan_tip" in wing:
        root = _kulfan_from_dict(wing["kulfan_root"])
        tip  = _kulfan_from_dict(wing["kulfan_tip"])
        return _lerp_kulfan(root, tip, eta)

    root_name = wing.get("airfoil_root", "naca0012")
    tip_name  = wing.get("airfoil_tip",  root_name)

    if root_name == tip_name:
        return root_name                               # same profile everywhere

    # Different named foils → convert and blend
    root = asb.Airfoil(root_name).to_kulfan_airfoil()
    tip  = asb.Airfoil(tip_name).to_kulfan_airfoil()
    return _lerp_kulfan(root, tip, eta)


def parse_yaml(path: Path) -> list:
    """
    Return a flat list of section dicts from a YAML wing definition.

    The y_half array defines the spanwise loft stations directly; no extra
    discretisation is needed.  Airfoil properties are evaluated (or
    interpolated) at each station.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    wing     = data["plane"]["wing"]
    y_half   = wing["y_half"]
    c_half   = wing["c_half"]
    xle_half = wing["xle_half"]
    twists   = wing.get("twist_half", [0.0] * len(y_half))
    dihedral = float(wing.get("dihedral", 0.0))
    y_tip    = float(max(y_half))
    ruled    = bool(wing.get("ruled", True))

    sections = []
    for i, y in enumerate(y_half):
        eta   = y / y_tip if y_tip > 0 else 0.0
        z_dih = y * np.sin(np.radians(dihedral))
        sections.append({
            "y":      y,
            "chord":  c_half[i],
            "xoff":   xle_half[i],
            "twist":  twists[i],
            "z_dih":  z_dih,
            "foil":   _foil_at_eta(wing, eta),
            "_ruled": ruled,
        })
    return sections


# ---------------------------------------------------------------------------
# XML input  (flow5 format, legacy)
# ---------------------------------------------------------------------------

def parse_xml(path: Path) -> list:
    """
    Read wing sections from a flow5 XML export.
    Dihedral is per-panel; integrates to absolute z positions.
    """
    root    = ET.parse(path).getroot()
    wing_el = root.find("wing")
    raw     = []
    for sec in wing_el.find("Sections").findall("Section"):
        raw.append({
            "y":        float(sec.findtext("y_position")),
            "chord":    float(sec.findtext("Chord")),
            "xoff":     float(sec.findtext("xOffset")),
            "dihedral": float(sec.findtext("Dihedral")),
            "twist":    float(sec.findtext("Twist")),
            "foil":     sec.findtext("Right_Side_FoilName"),
        })

    # Accumulate z from panel dihedral angles
    z = 0.0
    z_list = [0.0]
    for i in range(1, len(raw)):
        dih = np.radians(raw[i - 1]["dihedral"])
        dy  = raw[i]["y"] - raw[i - 1]["y"]
        z  += dy * np.sin(dih)
        z_list.append(z)

    return [
        {"y": r["y"], "chord": r["chord"], "xoff": r["xoff"],
         "twist": r["twist"], "z_dih": z_list[i], "foil": r["foil"]}
        for i, r in enumerate(raw)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert flow5 XML or YAML wing definition to STEP.")
    ap.add_argument("input",   type=Path, help="Input .xml or .yaml/.yml file")
    ap.add_argument("--out",   type=Path, default=None,
                    help="Output STEP path (default: same stem as input)")
    ap.add_argument("--half",  action="store_true",
                    help="Half-span only – skip Y=0 mirror")
    ap.add_argument("--n-pts", type=int, default=128,
                    help="Chordwise profile points per surface side (default 128)")
    ap.add_argument("--ruled", action=argparse.BooleanOptionalAction, default=None,
                    help="Ruled loft (linear between profiles). Overrides YAML setting.")
    args = ap.parse_args()

    ext = args.input.suffix.lower()
    if ext == ".xml":
        sections = parse_xml(args.input)
    elif ext in (".yaml", ".yml"):
        sections = parse_yaml(args.input)
    else:
        raise ValueError(f"Unsupported format: {ext}  (expected .xml or .yaml)")

    out_path = args.out or args.input.with_suffix(".step")
    y_tip    = max(s["y"] for s in sections)

    # ruled: CLI flag wins; fall back to YAML key; default True
    if args.ruled is not None:
        ruled = args.ruled
    else:
        ruled = bool(sections[0].get("_ruled", True))

    print(f"Wing: {len(sections)} spanwise stations")
    for s in sections:
        fd   = s["foil"]
        kind = fd if isinstance(fd, str) else \
               f"CST@η={s['y']/y_tip:.3f}"
        print(f"  y={s['y']:.4f}  c={s['chord']:.5f}  "
              f"xoff={s['xoff']:.5f}  twist={s['twist']:.2f}°  "
              f"z={s['z_dih']:.4f}  [{kind}]")

    print(f"\nBuilding {args.n_pts}-pt profiles, {'ruled' if ruled else 'smooth'} loft…")

    # Sections run root → tip (y ≥ 0).
    # For half-span: loft root → tip.
    # For full-span: loft negative-tip → root → positive-tip in one pass.
    # Doing it in one loft (rather than mirror+fuse) avoids a coincident-face
    # fuse that OpenCASCADE silently reduces to an empty solid.
    if args.half:
        wires = [
            _make_wire(s["foil"], s["chord"], s["xoff"],
                       s["y"], s["z_dih"], s["twist"], args.n_pts)
            for s in sections
        ]
        solid = cq.Solid.makeLoft(wires, ruled=ruled)
        print("Half-span solid created.")
    else:
        neg_wires = [
            _make_wire(s["foil"], s["chord"], s["xoff"],
                       -s["y"], s["z_dih"], s["twist"], args.n_pts)
            for s in reversed(sections)
        ]
        pos_wires = [
            _make_wire(s["foil"], s["chord"], s["xoff"],
                       s["y"], s["z_dih"], s["twist"], args.n_pts)
            for s in sections
        ]
        # neg_wires ends at root, pos_wires starts at root → deduplicate root wire
        all_wires = neg_wires + pos_wires[1:]
        solid = cq.Solid.makeLoft(all_wires, ruled=ruled)
        print(f"Full-span loft: {len(all_wires)} profiles spanning "
              f"−{sections[-1]['y']:.3f} m to +{sections[-1]['y']:.3f} m.")

    cq.exporters.export(cq.Workplane().add(solid), str(out_path))
    print(f"Exported → {out_path}")


if __name__ == "__main__":
    main()
