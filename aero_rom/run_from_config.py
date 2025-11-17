# CLI entry point

# run_from_config.py
import argparse
from src.pipeline import run_pipeline

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path to JSON config")
    args = ap.parse_args()
    info = run_pipeline(args.config)
    print("Models saved to:", info["models_dir"])
    print("Alpha bounds:", info["alpha_bounds"], "deg")
    print("Velocity bounds:", info["vel_bounds"], "m/s")
    print("\nUsage example:")
    print("""\
        from aero_rom.src.models import ClModel, CdModel, CmModel
        from aero_rom.src.interpolation import RGISurface

        cl = ClModel(RGISurface.from_npz("artifacts/models/cl_{partname}.npz"),
                    clip_alpha=%r, clip_vel=%r)
        cd = CdModel(RGISurface.from_npz("artifacts/models/cd_{partname}.npz"),
                    clip_alpha=%r, clip_vel=%r)
        cm = CmModel(RGISurface.from_npz("artifacts/models/cm_{partname}.npz"),
                    clip_alpha=%r, clip_vel=%r)

        a, V = 6.0, 18.0
        print("CL(a,V) =", cl(a,V))
        print("dCL/da, dCL/dV =", cl.backward(a,V))
        """ % (info["alpha_bounds"], info["vel_bounds"],
            info["alpha_bounds"], info["vel_bounds"],
            info["alpha_bounds"], info["vel_bounds"]))

if __name__ == "__main__":
    main()
