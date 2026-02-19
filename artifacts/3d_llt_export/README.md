# 3D LLT Artifacts Export

This directory contains exported 3D Lifting Line Theory (LLT) artifacts from the glider optimization pipeline, packaged for integration into 2D-only codebases.

## 📦 What's Included

Each timestamped export directory (e.g., `20260217_153219/`) contains:

### Wing Artifacts
- **Geometry files** (`wing_*.npy`): Spanwise discretization, chord distribution, twist, reference areas
- **Chebyshev coefficients** (`wing_cheby_phi_*.npy`): Reduced-order surrogate for CL, CD, CM as functions of (α, Re)

### Elevator Artifacts (if present)
- **Geometry files** (`elevator_*.npy`): Same structure as wing
- **Kulfan parameters** (`elevator_kulfan_*.npy`): Fixed airfoil shape representation
- **Chebyshev coefficients** (`elevator_cheby_phi_*.npy`): Reduced-order surrogate for CL_e, CD_e, CM_e

### Flow & Solver Parameters
- **Flow properties** (`flow_*.npy`): Air density (ρ), dynamic viscosity (μ)
- **Solver parameters** (`solver_*.npy`): LLT convergence settings (β, tol, iterations)

### Manifest
- **manifest.json**: Complete metadata including file names, shapes, dtypes, and usage instructions

## 🚀 Quick Start

### Load All Artifacts

```python
from example_load_artifacts import load_3d_llt_artifacts

# Load from timestamped directory
artifacts = load_3d_llt_artifacts("20260217_153219")

# Access wing geometry
wing_span = artifacts["wing_geometry"]["span"]  # meters
wing_area = artifacts["wing_geometry"]["S"]     # m²

# Access Chebyshev coefficients
phi_CL = artifacts["wing_chebyshev"]["phi_CL"]  # shape: (676, 1)
phi_CD = artifacts["wing_chebyshev"]["phi_CD"]
phi_CM = artifacts["wing_chebyshev"]["phi_CM"]
```

### Evaluate Surrogate Model

```python
from example_load_artifacts import evaluate_chebyshev_surrogate

# Define operating point
alpha = 5.0   # degrees
Re = 5000.0   # Reynolds number

# Define training ranges (must match export config)
alpha_range = (-30.0, 30.0)    # degrees
Re_range = (100.0, 100000.0)   # dimensionless

# Evaluate wing coefficients
wing_coeffs = evaluate_chebyshev_surrogate(
    alpha, Re,
    artifacts["wing_chebyshev"]["phi_CL"],
    artifacts["wing_chebyshev"]["phi_CD"],
    artifacts["wing_chebyshev"]["phi_CM"],
    alpha_range, Re_range
)

print(f"CL = {wing_coeffs['CL'][0]:.4f}")  # e.g., 0.2899
print(f"CD = {wing_coeffs['CD'][0]:.4f}")  # e.g., 0.0769
print(f"CM = {wing_coeffs['CM'][0]:.4f}")  # e.g., -0.0385
```

### Batch Evaluation

```python
import numpy as np

# Evaluate over a sweep
alphas = np.linspace(-10, 15, 50)
Re_constant = 5000.0

coeffs = evaluate_chebyshev_surrogate(
    alphas,
    np.full_like(alphas, Re_constant),
    artifacts["wing_chebyshev"]["phi_CL"],
    artifacts["wing_chebyshev"]["phi_CD"],
    artifacts["wing_chebyshev"]["phi_CM"],
    alpha_range, Re_range
)

# Plot polar
import matplotlib.pyplot as plt
plt.plot(coeffs["CD"], coeffs["CL"])
plt.xlabel("CD")
plt.ylabel("CL")
plt.title(f"3D LLT Polar at Re={Re_constant:.0f}")
plt.grid(True)
plt.show()
```

## 📐 Chebyshev Surrogate Model

The Chebyshev reduced-order model is a 2D polynomial approximation:

```
C(α, Re) = Σ_{i,j} φ_{ij} · T_i(α_scaled) · T_j(Re_scaled)
```

Where:
- **T_i, T_j**: Chebyshev polynomials of degree i, j (up to degree 25)
- **φ_{ij}**: Chebyshev coefficients (stored in `phi_CL`, `phi_CD`, `phi_CM`)
- **α_scaled, Re_scaled**: Input variables scaled to [-1, 1] domain

### Valid Input Ranges

**IMPORTANT**: The surrogate is only valid within the training ranges:

- **Angle of Attack**: -30° to +30°
- **Reynolds Number**: 100 to 100,000

Extrapolation outside these ranges will produce unreliable results!

## 🔧 Integration into 2D Codebases

### Option 1: Direct Coefficient Lookup

If your code expects simple 2D airfoil polars:

```python
# Replace your 2D lookup with Chebyshev evaluation
def get_wing_coeffs(alpha_deg, velocity, chord, rho, mu):
    Re = rho * velocity * chord / mu
    return evaluate_chebyshev_surrogate(
        alpha_deg, Re,
        wing_phi_CL, wing_phi_CD, wing_phi_CM,
        alpha_range, Re_range
    )
```

### Option 2: Pre-tabulate Polar

Generate a lookup table once at initialization:

```python
# Create 2D grid
alpha_grid = np.linspace(-30, 30, 100)
Re_grid = np.linspace(100, 100000, 50)
AA, RR = np.meshgrid(alpha_grid, Re_grid)

# Evaluate over grid
coeffs = evaluate_chebyshev_surrogate(
    AA.flatten(), RR.flatten(),
    wing_phi_CL, wing_phi_CD, wing_phi_CM,
    alpha_range, Re_range
)

# Reshape for interpolation
CL_table = coeffs["CL"].reshape(AA.shape)
CD_table = coeffs["CD"].reshape(AA.shape)
CM_table = coeffs["CM"].reshape(AA.shape)

# Use scipy.interpolate.RegularGridInterpolator for fast lookups
from scipy.interpolate import RegularGridInterpolator
CL_interp = RegularGridInterpolator((Re_grid, alpha_grid), CL_table)
```

## 📊 File Structure

```
artifacts/3d_llt_export/
├── example_load_artifacts.py      # Loading + evaluation utilities
├── README.md                       # This file
└── 20260217_153219/               # Timestamped export
    ├── manifest.json               # Metadata
    ├── wing_*.npy                  # Wing geometry (12 files)
    ├── wing_cheby_phi_*.npy        # Wing surrogates (3 files)
    ├── elevator_*.npy              # Elevator geometry (11 files)
    ├── elevator_kulfan_*.npy       # Elevator airfoil (4 files)
    ├── elevator_cheby_phi_*.npy    # Elevator surrogates (3 files)
    ├── flow_*.npy                  # Flow properties (2 files)
    └── solver_*.npy                # Solver params (5 files)
```

## ⚠️ Important Notes

1. **Coordinate System**: Right-handed, y+ points toward starboard (right wing)
2. **Units**: SI (meters, kg, Pa, etc.)
3. **Extrapolation**: Avoid evaluating outside training ranges!
4. **Chebyshev Basis**: Degree 25 → 676 coefficients per output (26×26 tensor product)
5. **Elevator**: Fixed airfoil (Kulfan params included), but α and Re can vary

## 🔍 Troubleshooting

### Large/Invalid Coefficient Values

**Symptom**: CL values like `42108435949500310362110004559872`

**Cause**: Evaluating outside training range (extrapolation)

**Fix**: Check that `alpha_range` and `Re_range` match the export config:
```python
# From conf/test.yaml
alpha_range = (-30.0, 30.0)
Re_range = (100.0, 100000.0)
```

### NaN Values

**Symptom**: `np.nan` in coefficient outputs

**Cause**: Division by zero in scaling (e.g., `Re_max == Re_min`)

**Fix**: Verify ranges are not degenerate intervals

### Shape Mismatches

**Symptom**: `ValueError: operands could not be broadcast together`

**Cause**: Input alpha/Re arrays have incompatible shapes

**Fix**: Ensure both are 1D arrays of same length, or broadcast correctly:
```python
alpha = np.array([5.0, 10.0, 15.0])
Re = np.array([5000.0, 5000.0, 5000.0])  # Must match alpha shape
```

## 📚 References

- **Lifting Line Theory**: Classical method for computing finite-wing effects
- **Chebyshev Polynomials**: Optimal basis for polynomial approximation on [-1,1]
- **3D Blocks Checkpoint**: `aero_rom/artifacts/models/3d_blocks.pt` (source geometry)

## 📧 Contact

For questions about these artifacts or integration help, contact the glider_optimization team.

---

**Export Date**: See `manifest.json` → `timestamp`  
**Export Config**: `conf/test.yaml` (see repository for full settings)
