#!/usr/bin/env python3
# run_from_config.py
#
# CLI entry point:
#   - builds full CL/CD/CM surfaces (wing & elevator) as CSVs
#   - builds differentiable 3D LLT blocks and saves them
#
# Usage:
#   python run_from_config.py data/config.json

import argparse
from src.diff_pipeline import run_pipeline as run_diff_pipeline


def main():
    ap = argparse.ArgumentParser(
        description="Run LLT + cuNeuralFoil to generate CSV surfaces and 3D differentiable blocks."
    )
    ap.add_argument("config", help="Path to JSON config")
    args = ap.parse_args()

    info = run_diff_pipeline(args.config)

    print("\n.csv raw surfaces saved to:", info["csv_dir"])
    print("3D differentiable blocks saved to:", info["models_dir"])
    print("Alpha bounds:", info["alpha_bounds"], "deg")
    print("Alpha steps:", info["alpha_steps"], "deg")
    print("Velocity bounds:", info["vel_bounds"], "m/s")
    print("Velocity steps:", info["vel_steps"], "m/s")
    print("Blocks checkpoint:", info["models_path"], "\n")


    print(
        'Example use:\n'
        '  import torch\n'
        '  ckpt = torch.load("artifacts/models/3d_blocks.pt", map_location="cuda")\n'
        '  cl_block_wing = ckpt["wing"]["cl_block"]\n'
        '  alpha, V = 5.0, 18.0\n'
        '  CL = cl_block_wing(alpha, V)\n'
        '  grads = cl_block_wing.backward(alpha, V)\n'
    )


if __name__ == "__main__":
    main()
