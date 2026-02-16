import pandas as pd
import numpy as np

print('=' * 80)
print('2D vs 3D Coefficient Comparison')
print('=' * 80)

# Load 2D data
df_2d = pd.read_csv('diagnostics/2026-02-13_3d-llt-debug/neuralfoil_2d_raw_outputs.csv')
# Load 3D data  
df_3d = pd.read_csv('diagnostics/2026-02-13_3d-llt-debug/llt_wing_raw_outputs.csv')

print(f'\n📊 2D NeuralFoil (use_3d_llt: false):')
print(f'   CL: [{df_2d.CL.min():.3f}, {df_2d.CL.max():.3f}], mean={df_2d.CL.mean():.3f}')
print(f'   CD: [{df_2d.CD.min():.3f}, {df_2d.CD.max():.3f}], mean={df_2d.CD.mean():.3f}')
print(f'   CM: [{df_2d.CM.min():.3f}, {df_2d.CM.max():.3f}], mean={df_2d.CM.mean():.3f}')

print(f'\n📊 3D LLT (use_3d_llt: true):')
print(f'   CL: [{df_3d.CL.min():.3f}, {df_3d.CL.max():.3f}], mean={df_3d.CL.mean():.3f}')
print(f'   CD: [{df_3d.CD.min():.3f}, {df_3d.CD.max():.3f}], mean={df_3d.CD.mean():.3f}')
print(f'   CM: [{df_3d.CM.min():.3f}, {df_3d.CM.max():.3f}], mean={df_3d.CM.mean():.3f}')

print(f'\n📈 Difference (3D - 2D):')
print(f'   CL max: {df_3d.CL.max() - df_2d.CL.max():.3f} ({(df_3d.CL.max() / df_2d.CL.max() - 1)*100:.1f}% higher)')
print(f'   CD max: {df_3d.CD.max() - df_2d.CD.max():.3f} ({(df_3d.CD.max() / df_2d.CD.max() - 1)*100:.1f}% higher)')
print(f'   CL mean: {df_3d.CL.mean() - df_2d.CL.mean():.3f}')

print(f'\n✅ 2D: Optimization SUCCEEDS (obj=15.766)')
print(f'❌ 3D: Optimization FAILS (NaN at constraint 212, variable 155)')
print('\n' + '=' * 80)
