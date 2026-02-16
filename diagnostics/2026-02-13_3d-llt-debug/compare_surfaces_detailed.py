import pandas as pd
import numpy as np

# Load the raw data
df_2d = pd.read_csv('neuralfoil_2d_raw_outputs.csv')
df_3d = pd.read_csv('llt_3d_wing_raw_outputs.csv')

# Merge on alpha and Re to get paired comparisons
merged = pd.merge(
    df_2d, df_3d, 
    on=['alpha_deg', 'Re'], 
    suffixes=('_2d', '_3d')
)

print("=== Point-by-Point Differences ===")
print(f"\nCL differences:")
print(f"  Mean absolute diff: {np.abs(merged['CL_3d'] - merged['CL_2d']).mean():.6f}")
print(f"  Max absolute diff: {np.abs(merged['CL_3d'] - merged['CL_2d']).max():.6f}")
print(f"  RMS diff: {np.sqrt(((merged['CL_3d'] - merged['CL_2d'])**2).mean()):.6f}")

print(f"\nCD differences:")
print(f"  Mean absolute diff: {np.abs(merged['CD_3d'] - merged['CD_2d']).mean():.6f}")
print(f"  Max absolute diff: {np.abs(merged['CD_3d'] - merged['CD_2d']).max():.6f}")

print(f"\n=== Where are the biggest differences? ===")
merged['CL_diff'] = np.abs(merged['CL_3d'] - merged['CL_2d'])
biggest = merged.nlargest(10, 'CL_diff')[['alpha_deg', 'Re', 'CL_2d', 'CL_3d', 'CL_diff']]
print(biggest.to_string(index=False))

# Check gradients (finite differences)
print(f"\n=== Surface Gradient Comparison ===")
# Sort by alpha, Re
df_2d_sorted = df_2d.sort_values(['Re', 'alpha_deg'])
df_3d_sorted = df_3d.sort_values(['Re', 'alpha_deg'])

# Compute dCL/dAlpha
dCL_dalpha_2d = np.diff(df_2d_sorted['CL'].values)
dCL_dalpha_3d = np.diff(df_3d_sorted['CL'].values)

print(f"2D gradient range: [{dCL_dalpha_2d.min():.4f}, {dCL_dalpha_2d.max():.4f}]")
print(f"3D gradient range: [{dCL_dalpha_3d.min():.4f}, {dCL_dalpha_3d.max():.4f}]")

# Check for discontinuities (large jumps)
print(f"\n2D max gradient jump: {np.abs(dCL_dalpha_2d).max():.4f}")
print(f"3D max gradient jump: {np.abs(dCL_dalpha_3d).max():.4f}")
