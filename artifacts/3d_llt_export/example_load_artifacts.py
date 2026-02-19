"""
Example: Loading 3D LLT artifacts exported from glider_optimization

This script demonstrates how to load and use the .npy files exported
from the 3D LLT pipeline for integration into a 2D-only codebase.
"""

import numpy as np
from pathlib import Path
import json


def load_3d_llt_artifacts(export_dir):
    """
    Load all 3D LLT artifacts from an export directory.
    
    Args:
        export_dir: Path to the timestamped export directory
                    (e.g., "artifacts/3d_llt_export/20260217_153219")
    
    Returns:
        dict: Dictionary containing all loaded artifacts organized by category
    """
    export_path = Path(export_dir)
    
    # Load manifest
    with open(export_path / "manifest.json", "r") as f:
        manifest = json.load(f)
    
    print(f"Loading artifacts from: {export_dir}")
    print(f"Export timestamp: {manifest['timestamp']}")
    print()
    
    # Load wing geometry
    wing_geom = {}
    print("📐 Loading wing geometry...")
    for key, info in manifest["wing_geometry"].items():
        filepath = export_path / info["filename"]
        wing_geom[key] = np.load(filepath)
        print(f"  {key:15s}: shape={wing_geom[key].shape}, dtype={wing_geom[key].dtype}")
    
    # Load wing Chebyshev coefficients
    wing_cheby = {}
    print("\n🔢 Loading wing Chebyshev coefficients...")
    for key, info in manifest["wing_chebyshev"].items():
        filepath = export_path / info["filename"]
        wing_cheby[key] = np.load(filepath)
        print(f"  {key:15s}: shape={wing_cheby[key].shape}, range=[{wing_cheby[key].min():.3e}, {wing_cheby[key].max():.3e}]")
    
    # Load flow properties
    flow = {}
    print("\n💨 Loading flow properties...")
    for key, info in manifest["flow_properties"].items():
        filepath = export_path / info["filename"]
        flow[key] = np.load(filepath).item()  # Scalars need .item() to extract value
        print(f"  {key:15s}: {flow[key]:.6e}")
    
    # Load solver parameters
    solver = {}
    print("\n⚙️  Loading solver parameters...")
    for key, info in manifest["solver_parameters"].items():
        filepath = export_path / info["filename"]
        solver[key] = np.load(filepath).item()
        print(f"  {key:15s}: {solver[key]}")
    
    # Load elevator (if present)
    elevator_geom = {}
    elevator_kulfan = {}
    elevator_cheby = {}
    if manifest["elevator"]:
        print("\n🛩️  Loading elevator geometry...")
        for key, info in manifest["elevator"].items():
            filepath = export_path / info["filename"]
            value = np.load(filepath)
            
            if "kulfan" in key:
                k = key.replace("kulfan_", "")
                elevator_kulfan[k] = value.item() if value.ndim == 0 else value
                print(f"  Kulfan {k:15s}: shape={elevator_kulfan[k].shape if hasattr(elevator_kulfan[k], 'shape') else 'scalar'}")
            elif "cheby" in key:
                k = key.replace("cheby_", "")
                elevator_cheby[k] = value
                print(f"  Cheby  {k:15s}: shape={elevator_cheby[k].shape}, range=[{value.min():.3e}, {value.max():.3e}]")
            else:
                elevator_geom[key] = value.item() if value.ndim == 0 else value
                print(f"  Geom   {key:15s}: shape={elevator_geom[key].shape if hasattr(elevator_geom[key], 'shape') else 'scalar'}")
    
    return {
        "wing_geometry": wing_geom,
        "wing_chebyshev": wing_cheby,
        "flow_properties": flow,
        "solver_parameters": solver,
        "elevator_geometry": elevator_geom if elevator_geom else None,
        "elevator_kulfan": elevator_kulfan if elevator_kulfan else None,
        "elevator_chebyshev": elevator_cheby if elevator_cheby else None,
        "manifest": manifest
    }


def evaluate_chebyshev_surrogate(alpha_deg, Re, phi_CL, phi_CD, phi_CM, 
                                   alpha_range, Re_range, degree=25):
    """
    Evaluate Chebyshev surrogate model for aerodynamic coefficients.
    
    Args:
        alpha_deg: Angle of attack in degrees (scalar or array)
        Re: Reynolds number (scalar or array)
        phi_CL, phi_CD, phi_CM: Chebyshev coefficient arrays (shape: [(degree+1)^2, 1])
        alpha_range: (min, max) for alpha normalization
        Re_range: (min, max) for Re normalization
        degree: Chebyshev polynomial degree (default: 25)
    
    Returns:
        dict: {"CL": ..., "CD": ..., "CM": ...}
    """
    # Scale to [-1, 1] domain
    alpha_scaled = 2 * (alpha_deg - alpha_range[0]) / (alpha_range[1] - alpha_range[0]) - 1
    Re_scaled = 2 * (Re - Re_range[0]) / (Re_range[1] - Re_range[0]) - 1
    
    # Compute Chebyshev basis functions
    def chebyshev_basis_1d(x, degree):
        """Compute 1D Chebyshev basis up to degree."""
        T = np.zeros((x.size if hasattr(x, 'size') else 1, degree + 1))
        T[:, 0] = 1
        if degree >= 1:
            T[:, 1] = x if hasattr(x, '__len__') else np.array([x])
        for n in range(2, degree + 1):
            T[:, n] = 2 * (x if hasattr(x, '__len__') else np.array([x])) * T[:, n-1] - T[:, n-2]
        return T
    
    T_alpha = chebyshev_basis_1d(alpha_scaled, degree)
    T_Re = chebyshev_basis_1d(Re_scaled, degree)
    
    # Tensor product basis
    X = (T_alpha[:, :, None] * T_Re[:, None, :]).reshape(T_alpha.shape[0], -1)
    
    # Evaluate
    CL = X @ phi_CL
    CD = X @ phi_CD
    CM = X @ phi_CM
    
    return {"CL": CL.flatten(), "CD": CD.flatten(), "CM": CM.flatten()}


if __name__ == "__main__":
    # Example usage
    export_dir = "artifacts/3d_llt_export/20260217_153219"
    
    # Load all artifacts
    artifacts = load_3d_llt_artifacts(export_dir)
    
    print("\n" + "="*80)
    print("✅ Artifacts loaded successfully!")
    print("="*80)
    
    # Example: Evaluate wing surrogate at a test point
    print("\n🧪 Testing wing surrogate evaluation...")
    test_alpha = 5.0  # degrees
    test_Re = 5000.0  # Within the training range
    
    # Get alpha/Re ranges from manifest (should match training config)
    # From conf/test.yaml: AoA_min: -30, AoA_max: 30, Re_min: 100, Re_max: 100000
    alpha_range = (-30.0, 30.0)  # degrees
    Re_range = (100.0, 100000.0)
    
    wing_coeffs = evaluate_chebyshev_surrogate(
        test_alpha, test_Re,
        artifacts["wing_chebyshev"]["phi_CL"],
        artifacts["wing_chebyshev"]["phi_CD"],
        artifacts["wing_chebyshev"]["phi_CM"],
        alpha_range, Re_range
    )
    
    print(f"Test point: α={test_alpha}°, Re={test_Re:.0f}")
    print(f"Wing CL: {wing_coeffs['CL'][0]:.4f}")
    print(f"Wing CD: {wing_coeffs['CD'][0]:.4f}")
    print(f"Wing CM: {wing_coeffs['CM'][0]:.4f}")
    
    # Example: Evaluate elevator surrogate (if available)
    if artifacts["elevator_chebyshev"]:
        print("\n🧪 Testing elevator surrogate evaluation...")
        elevator_coeffs = evaluate_chebyshev_surrogate(
            test_alpha, test_Re,
            artifacts["elevator_chebyshev"]["phi_CL"],
            artifacts["elevator_chebyshev"]["phi_CD"],
            artifacts["elevator_chebyshev"]["phi_CM"],
            alpha_range, Re_range
        )
        
        print(f"Test point: α={test_alpha}°, Re={test_Re:.0f}")
        print(f"Elevator CL: {elevator_coeffs['CL'][0]:.4f}")
        print(f"Elevator CD: {elevator_coeffs['CD'][0]:.4f}")
        print(f"Elevator CM: {elevator_coeffs['CM'][0]:.4f}")
    
    print("\n" + "="*80)
    print("📦 Artifact Summary:")
    print(f"  Wing panels: {len(artifacts['wing_geometry']['y_mid'])}")
    print(f"  Wing span: {artifacts['wing_geometry']['span']:.3f} m")
    print(f"  Wing area: {artifacts['wing_geometry']['S']:.3f} m²")
    if artifacts["elevator_geometry"]:
        print(f"  Elevator span: {artifacts['elevator_geometry']['span']:.3f} m")
        print(f"  Elevator area: {artifacts['elevator_geometry']['S']:.3f} m²")
    print(f"  Chebyshev degree: 25 (676 coefficients per output)")
    print(f"  Air density: {artifacts['flow_properties']['rho']:.4f} kg/m³")
    print(f"  Air viscosity: {artifacts['flow_properties']['mu']:.6e} Pa·s")
    print("="*80)
