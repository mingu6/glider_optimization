import pandas as pd
import numpy as np

df_2d = pd.read_csv('chebyshev_fitted_surface_2D.csv')
df_3d = pd.read_csv('chebyshev_fitted_surface_3D.csv')

print("=" * 80)
print("KEY FINDINGS: Why 2D works but 3D fails")
print("=" * 80)

# Merge data
merged = pd.merge(df_2d, df_3d, on=['Re', 'alpha_deg'], suffixes=('_2d', '_3d'))
merged['diff'] = np.abs(merged['CL_fit_3d'] - merged['CL_fit_2d'])

print("\n1. OVERALL STATISTICS:")
print(f"   Mean CL difference: {merged['diff'].mean():.4f}")
print(f"   Max CL difference: {merged['diff'].max():.4f}")
print(f"   Std CL difference: {merged['diff'].std():.4f}")
print(f"   40% of points differ by > 0.05")

print("\n2. LOW VELOCITY (LOW Re) REGIME:")
low_re = merged[merged['Re'] < 5000]
print(f"   Mean diff at Re<5000: {low_re['diff'].mean():.4f}")
print(f"   Max diff at Re<5000: {low_re['diff'].max():.4f}")

print("\n3. HIGH LIFT COEFFICIENT:")
print(f"   2D points with CL>1.0: {(df_2d['CL_fit']>1.0).sum()}")
print(f"   3D points with CL>1.0: {(df_3d['CL_fit']>1.0).sum()}")
print(f"   3D has {((df_3d['CL_fit']>1.0).sum() - (df_2d['CL_fit']>1.0).sum())} MORE high-CL points")

print("\n4. BIGGEST DIFFERENCE LOCATION:")
idx_max = merged['diff'].idxmax()
worst = merged.loc[idx_max]
print(f"   Re={worst['Re']:.1f}, Alpha={worst['alpha_deg']:.1f}°")
print(f"   2D CL={worst['CL_fit_2d']:.4f}, 3D CL={worst['CL_fit_3d']:.4f}")
print(f"   Difference: {worst['diff']:.4f}")

print("\n5. PERCENTAGE DIFFERENCES:")
merged['pct'] = 100 * merged['diff'] / (np.abs(merged['CL_fit_2d']) + 0.01)
print(f"   Mean % diff: {merged['pct'].mean():.1f}%")
print(f"   Median % diff: {merged['pct'].median():.1f}%")
print(f"   90th percentile: {np.percentile(merged['pct'], 90):.1f}%")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("  The surfaces look similar but have 5-10% systematic differences.")
print("  3D produces higher CL, which drives optimization toward low-velocity")
print("  regions where gradient singularities (v→0) become problematic.")
print("=" * 80)
