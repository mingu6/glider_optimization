# end-to-end: generate CSV -> train -> save

import numpy as np
from pathlib import Path
from src.config import load_config
from src.io_utils import save_surface_csv
from src.interpolation import RGISurface
from src.llt import run_llt
from src.llt import LLT_computational_params as compute_llt_params




def run_pipeline(config_path: str):


    cfg = load_config(config_path)
    outdir = Path(cfg.get("output_dir", "artifacts"))
    (outdir / "raw_surfaces").mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    alphas = np.arange(cfg['flow']['alpha_range'][0], cfg['flow']['alpha_range'][1]+cfg['flow']['alpha_step'], cfg['flow']['alpha_step'])
    vels   = np.arange(cfg['flow']['vel_range'][0], cfg['flow']['vel_range'][1]+cfg['flow']['vel_step'], cfg['flow']['vel_step'])

    R = np.array(cfg['flow']["R"], dtype=float)  # specific gas constant for air
    T = np.array(cfg['flow']["T"], dtype=float)  # temperature in K
    p = np.array(cfg['flow']["p"], dtype=float)  # pressure in Pa
    rho = np.array(cfg['flow']["rho"], dtype=float)  # density in kg/m3
    nu = np.array(cfg['flow']["nu"], dtype=float)  # kinematic viscosity in m2/s
    mu = rho * nu  # dynamic viscosity

    airflow={'R':R,'T':T,'p':p,'rho':rho,'nu':nu, 'mu':mu}

    # Produce surfaces
    # Wing
    computation_params_wing= compute_llt_params(cfg["wing_geometry"]["y_half"], cfg["wing_geometry"]["c_half"], 
                                                cfg["wing_geometry"]["xle_half"], cfg["wing_geometry"]["twist_half"], 
                                                cfg["wing_geometry"]["airfoil"])
    
    df_wing = run_llt(computation_params_wing['airfoil_CST'], alphas, vels, airflow, computation_params_wing)

    alpha_vals_wing = np.sort(df_wing["AoA"].unique())
    vel_vals_wing   = np.sort(df_wing["V_inf"].unique())
    CL_grid_wing = df_wing.pivot(index="AoA", columns="V_inf", values="CL").values
    CD_grid_wing = df_wing.pivot(index="AoA", columns="V_inf", values="CD").values
    CM_grid_wing = df_wing.pivot(index="AoA", columns="V_inf", values="CM_pitch").values

    # Elevator
    computation_params_elevator= compute_llt_params(cfg["elevator_geometry"]["y_half"], cfg["elevator_geometry"]["c_half"], 
                                                   cfg["elevator_geometry"]["xle_half"], cfg["elevator_geometry"]["twist_half"], 
                                                   cfg["elevator_geometry"]["airfoil"])
    
    df_elevator = run_llt(computation_params_elevator['airfoil_CST'], alphas, vels, airflow, computation_params_elevator)  

    alpha_vals_elevator = np.sort(df_elevator["AoA"].unique())
    vel_vals_elevator   = np.sort(df_elevator["V_inf"].unique())
    CL_grid_elevator = df_elevator.pivot(index="AoA", columns="V_inf", values="CL").values
    CD_grid_elevator = df_elevator.pivot(index="AoA", columns="V_inf", values="CD").values
    CM_grid_elevator = df_elevator.pivot(index="AoA", columns="V_inf", values="CM_pitch").values

    # Backup CSVs
    save_surface_csv(outdir/"raw_surfaces/cl_wing.csv", alpha_vals_wing, vel_vals_wing, CL_grid_wing, "CL")
    save_surface_csv(outdir/"raw_surfaces/cd_wing.csv", alpha_vals_wing, vel_vals_wing, CD_grid_wing, "CD")
    save_surface_csv(outdir/"raw_surfaces/cm_wing.csv", alpha_vals_wing, vel_vals_wing, CM_grid_wing, "CM")
    save_surface_csv(outdir/"raw_surfaces/cl_elevator.csv", alpha_vals_elevator, vel_vals_elevator, CL_grid_elevator, "CL")
    save_surface_csv(outdir/"raw_surfaces/cd_elevator.csv", alpha_vals_elevator, vel_vals_elevator, CD_grid_elevator, "CD")
    save_surface_csv(outdir/"raw_surfaces/cm_elevator.csv", alpha_vals_elevator, vel_vals_elevator, CM_grid_elevator, "CM")

    # Train RGI cubic & save
    RGISurface(alpha_vals_wing, vel_vals_wing, CL_grid_wing).to_npz(outdir/"models/cl_wing.npz")
    RGISurface(alpha_vals_wing, vel_vals_wing, CD_grid_wing).to_npz(outdir/"models/cd_wing.npz")
    RGISurface(alpha_vals_wing, vel_vals_wing, CM_grid_wing).to_npz(outdir/"models/cm_wing.npz")
    RGISurface(alpha_vals_elevator, vel_vals_elevator, CL_grid_elevator).to_npz(outdir/"models/cl_elevator.npz")
    RGISurface(alpha_vals_elevator, vel_vals_elevator, CD_grid_elevator).to_npz(outdir/"models/cd_elevator.npz")
    RGISurface(alpha_vals_elevator, vel_vals_elevator, CM_grid_elevator).to_npz(outdir/"models/cm_elevator.npz")

    return {
        "alpha_bounds": (float(alphas.min()), float(alphas.max())),
        "vel_bounds":   (float(vels.min()), float(vels.max())),
        "models_dir":   str(outdir/"models")
    }
