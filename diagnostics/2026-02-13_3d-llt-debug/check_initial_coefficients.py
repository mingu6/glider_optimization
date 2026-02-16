"""
FINAL DIAGNOSIS: Print the actual Chebyshev coefficients and understand
why 3D causes NaN but 2D doesn't at the INITIAL EVALUATION.

Key insight: NaN happens BEFORE iteration 0, meaning it's in the
initial constraint/gradient evaluation, NOT during optimization.
"""

import pandas as pd
import numpy as np

print("="*80)
print("COMPARING 3D vs 2D CHEBYSHEV COEFFICIENTS")
print("="*80)

# Load the Chebyshev coefficients from the fitted surfaces
gt_2d = pd.read_csv('diagnostics/2026-02-13_3d-llt-debug/chebyshev_ground_truth_2D.csv')
gt_3d = pd.read_csv('diagnostics/2026-02-13_3d-llt-debug/chebyshev_ground_truth_3D.csv')

print("\n📊 At Initial Guess Conditions:")
print("   vx = 7.0 m/s, vz = 0.0 m/s")
print("   v = 7.0 m/s")
print("   theta = 0°, alpha = 0°")
print("   chord ~ 0.15 m")
print("   Re = rho * v * c / mu = 1.225 * 7 * 0.15 / 1.789e-5 ≈ 71,900")

# Find coefficients near this operating point
re_target = 71900
alpha_target = 0.0

# Find closest point in the data
gt_2d['dist'] = np.sqrt((gt_2d['Re'] - re_target)**2 + (gt_2d['alpha_deg'] - alpha_target)**2)
gt_3d['dist'] = np.sqrt((gt_3d['Re'] - re_target)**2 + (gt_3d['alpha_deg'] - alpha_target)**2)

closest_2d = gt_2d.loc[gt_2d['dist'].idxmin()]
closest_3d = gt_3d.loc[gt_3d['dist'].idxmin()]

print(f"\n📍 Closest 2D point: Re={closest_2d['Re']:.0f}, alpha={closest_2d['alpha_deg']:.2f}°")
print(f"   CL={closest_2d['CL']:.4f}, CD={closest_2d['CD']:.4f}, CM={closest_2d['CM']:.4f}")

print(f"\n📍 Closest 3D point: Re={closest_3d['Re']:.0f}, alpha={closest_3d['alpha_deg']:.2f}°")
print(f"   CL={closest_3d['CL']:.4f}, CD={closest_3d['CD']:.4f}, CM={closest_3d['CM']:.4f}")

print(f"\n🔍 Difference at initial conditions:")
print(f"   ΔCL = {closest_3d['CL'] - closest_2d['CL']:.4f}")
print(f"   ΔCD = {closest_3d['CD'] - closest_2d['CD']:.4f}")
print(f"   ΔCM = {closest_3d['CM'] - closest_2d['CM']:.4f}")

# Check for extreme values that could cause issues
print("\n🔍 Extreme Values Check:")
print("2D:")
print(f"   CL: min={gt_2d['CL'].min():.4f}, max={gt_2d['CL'].max():.4f}")
print(f"   CD: min={gt_2d['CD'].min():.4f}, max={gt_2d['CD'].max():.4f}")
print(f"   CM: min={gt_2d['CM'].min():.4f}, max={gt_2d['CM'].max():.4f}")

print("3D:")
print(f"   CL: min={gt_3d['CL'].min():.4f}, max={gt_3d['CL'].max():.4f}")
print(f"   CD: min={gt_3d['CD'].min():.4f}, max={gt_3d['CD'].max():.4f}")
print(f"   CM: min={gt_3d['CM'].min():.4f}, max={gt_3d['CM'].max():.4f}")

print("\n" + "="*80)
print("HYPOTHESIS:")
print("="*80)
print("""
The NaN occurs at INITIAL EVALUATION, not during optimization.
This means the initial guess (vx=7, vz=0, alpha=0, Re~72k) causes
a problem when evaluating either:

1. The dynamics constraints at the initial point
2. The gradients of those constraints
3. The Chebyshev polynomial basis evaluation

Since 2D and 3D have similar values at this operating point,
the issue is likely in:
- Higher-order polynomial terms in 3D being more sensitive
- Numerical conditioning of the Chebyshev basis at this Re/alpha
- A specific term in the dynamics that becomes singular

Next step: Instrument glider_jinenv.py to log the actual values
being computed when the dynamics are evaluated at the initial point.
""")
