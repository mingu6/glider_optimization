#!/usr/bin/env python3
# run_from_config.py
#
# CLI entry point:
#   - builds full CL/CD/CM surfaces (wing & elevator) as CSVs
#   - builds differentiable 3D LLT blocks and saves them
#
# Usage:
#   python run_from_config.py data/config.json [--export_surfaces] [--no-export_ckpt]
#
# Examples:
#   python run_from_config.py data/config.json                    # checkpoint only (default)
#   python run_from_config.py data/config.json --export_surfaces  # checkpoint + CSVs
#   python run_from_config.py data/config.json --no-export_ckpt   # CSVs only (no checkpoint)

import argparse
from src.diff_pipeline import run_pipeline as run_diff_pipeline


def main():
    ap = argparse.ArgumentParser(
        description="Run LLT + cuNeuralFoil to generate CSV surfaces and 3D differentiable blocks."
    )
    ap.add_argument("config", help="Path to JSON config")
    ap.add_argument(
        "--export_surfaces",
        action="store_true",
        help="Export CSV coefficient surfaces (default: False)"
    )
    ap.add_argument(
        "--no-export_ckpt",
        action="store_false",
        dest="export_ckpt",
        help="Skip checkpoint export (default: exports checkpoint)"
    )
    args = ap.parse_args()

    info = run_diff_pipeline(
        args.config,
        export_surfaces=args.export_surfaces,
        export_ckpt=args.export_ckpt
    )

    print("\n" + "="*60)
    print("Pipeline Complete")
    print("="*60)
    if info["csv_dir"]:
        print("CSV surfaces saved to:", info["csv_dir"])
    if info["models_path"]:
        print("Checkpoint saved to:", info["models_path"])
    print("Alpha bounds:", info["alpha_bounds"], "deg")
    print("Alpha steps:", info["alpha_steps"], "deg")
    print("Velocity bounds:", info["vel_bounds"], "m/s")
    print("Velocity steps:", info["vel_steps"], "m/s")
    
    if info["models_path"]:
        print("\n" + "="*60)
        print("Usage Example")
        print("="*60)
        print("from src.aero_block import AeroBlock")
        print(f"aero = AeroBlock.from_ckpt('{info['models_path']}', part='wing', mode='implicit')")
        print("coeffs = aero.forward(alpha=5.0, V=18.0)")
        print("grads = aero.backward_combined(alpha=5.0, V=18.0, v_CL=1.0, v_CD=0.0, v_CM=0.0)")
    print("="*60 + "\n")



if __name__ == "__main__":
    main()
