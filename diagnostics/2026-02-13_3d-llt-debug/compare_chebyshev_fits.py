import pandas as pd
import numpy as np

# Load fitted surfaces (100x100 grid)
df_2d_fit = pd.read_csv('chebyshev_fitted_surface_2D.csv')
df_3d_fit = pd.read_csv('chebyshev_fitted_surface_3D.csv')

# Merge on Re and alpha
merged = pd.merge(
    df_2d_fit, df_3d_fit,
    on=['Re', 'alpha_deg'],
    suffixes=('_2d', '_3d')
)

print("=== Chebyshev Fitted Surface Differences ===")
print(f"\nCL_fit differences:")
print(f"  Mean: {(merged['CL_fit_3d'] - merged['CL_fit_2d']).mean():.6f}")
print(f"  Std: {(merged['CL_fit_3d'] - merged['CL_fit_2d']).std():.6f}")
print(f"  Max absolute: {np.abs(merged['CL_fit_3d'] - merged['CL_fit_2d']).max():.6f}")

# Check for regions with large differences
merged['CL_diff'] = np.abs(merged['CL_fit_3d'] - merged['CL_fit_2d'])
print(f"\n=== Regions with >0.05 CL difference ===")
large_diff = merged[merged['CL_diff'] > 0.05]
print(f"Number of points: {len(large_diff)} / {len(merged)}")
if len(large_diff) > 0:
    print(large_diff[['alpha_deg', 'Re', 'CL_fit_2d', 'CL_fit_3d', 'CL_diff']].head(20).to_string(index=False))

# Check second derivatives (curvature)
print(f"\n=== Surface Curvature (d2CL/dAlpha2) ===")
# Simple finite difference approximation
for Re_val in [1000, 10000, 50000]:
    subset = merged[np.isclose(merged['Re'], Re_val, rtol=0.05)]
    if len(subset) > 3:
        subset = subset.sort_values('alpha_deg')
        d2CL_2d = np.diff(np.diff(subset['CL_fit_2d'].values))
        d2CL_3d = np.diff(np.diff(subset['CL_fit_3d'].values))
        print(f"Re={Re_val:.0f}: 2D curvature range [{d2CL_2d.min():.6f}, {d2CL_2d.max():.6f}]")
        print(f"Re={Re_val:.0f}: 3D curvature range [{d2CL_3d.min():.6f}, {d2CL_3d.max():.6f}]")

# Check derivative w.r.t Reynolds
print(f"\n=== dCL/dRe Comparison ===")
for alpha_val in [-20, 0, 20]:
    subset = merged[np.isclose(merged['alpha_deg'], alpha_val, atol=1.0)]
    if len(subset) > 3:
        subset = subset.sort_values('Re')
        dCL_dRe_2d = np.diff(subset['CL_fit_2d'].values) / np.diff(subset['Re'].values)
        dCL_dRe_3d = np.diff(subset['CL_fit_3d'].values) / np.diff(subset['Re'].values)
        print(f"Alpha={alpha_val:.0f}°: 2D dCL/dRe range [{dCL_dRe_2d.min():.8f}, {dCL_dRe_2d.max():.8f}]")
        print(f"Alpha={alpha_val:.0f}°: 3D dCL/dRe range [{dCL_dRe_3d.min():.8f}, {dCL_dRe_3d.max():.8f}]")

# Percentage difference
print(f"\n=== Percentage Difference ===")
merged['CL_pct_diff'] = 100 * np.abs(merged['CL_fit_3d'] - merged['CL_fit_2d']) / (np.abs(merged['CL_fit_2d']) + 1e-10)
print(f"Mean percentage diff: {merged['CL_pct_diff'].mean():.2f}%")
print(f"Median percentage diff: {merged['CL_pct_diff'].median():.2f}%")
print(f"Max percentage diff: {merged['CL_pct_diff'].max():.2f}%")
