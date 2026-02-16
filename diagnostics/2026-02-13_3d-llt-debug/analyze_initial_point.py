"""
Extract and analyze the INITIAL POINT that causes NaN.

Since NaN occurs during initial evaluation (before iteration 0),
we need to understand what initial guess IPOPT is given.
"""

import numpy as np
import pickle
from pathlib import Path

def decode_decision_variables(x, n_stages=111):
    """
    Decode flattened decision variable vector into states and controls.
    
    Structure (from go_safe_pdp.py):
    - States: 6 per stage (x, z, theta, phi, vx, vz) at stages 0..111 (112 total)
    - Controls: 1 per stage (elevator) at stages 0..110 (111 total)
    
    Total variables: 6*112 + 1*111 = 672 + 111 = 783
    """
    n_states_per_stage = 6
    n_controls_per_stage = 1
    
    # Extract states (stages 0 to n_stages)
    n_state_vars = n_states_per_stage * (n_stages + 1)
    states = x[:n_state_vars].reshape((n_stages + 1), n_states_per_stage)
    
    # Extract controls (stages 0 to n_stages-1)
    controls = x[n_state_vars:].reshape(n_stages, n_controls_per_stage)
    
    return states, controls


def find_problematic_values(states, controls):
    """Find values that could cause NaN in gradients."""
    
    print("\n" + "="*80)
    print("ANALYZING INITIAL POINT FOR NaN SOURCES")
    print("="*80)
    
    # Variable 155 analysis
    print("\n🎯 Variable 155 Analysis:")
    print("   Layout: [state_stage_0 (6 vars), state_stage_1 (6 vars), ...]")
    print(f"   Variable 155 = state variable at index 155")
    
    # Determine which state and stage
    state_idx_in_flat = 155
    stage = state_idx_in_flat // 6
    component = state_idx_in_flat % 6
    
    state_names = ['x', 'z', 'theta', 'phi', 'vx', 'vz']
    print(f"   Stage: {stage}/111")
    print(f"   Component: {component} ({state_names[component]})")
    print(f"   Value: {states[stage, component]:.6f}")
    
    # Check for problematic velocities
    print("\n🔍 Velocity Analysis:")
    vx = states[:, 4]  # x-velocity
    vz = states[:, 5]  # z-velocity
    v_mag = np.sqrt(vx**2 + vz**2)
    
    print(f"   vx range: [{vx.min():.4f}, {vx.max():.4f}]")
    print(f"   vz range: [{vz.min():.4f}, {vz.max():.4f}]")
    print(f"   |v| range: [{v_mag.min():.4f}, {v_mag.max():.4f}]")
    
    # Check for near-zero velocities
    v_floor = 0.1
    near_zero = v_mag < v_floor
    if near_zero.any():
        print(f"\n   ⚠️ {near_zero.sum()} stages have |v| < {v_floor}")
        indices = np.where(near_zero)[0]
        print(f"   Stages: {indices[:10]}")  # Show first 10
        for idx in indices[:5]:
            print(f"      Stage {idx}: vx={vx[idx]:.6f}, vz={vz[idx]:.6f}, |v|={v_mag[idx]:.6f}")
    
    # Check for NaN or Inf
    print(f"\n   States contain NaN: {np.isnan(states).any()}")
    print(f"   States contain Inf: {np.isinf(states).any()}")
    print(f"   Controls contain NaN: {np.isnan(controls).any()}")
    print(f"   Controls contain Inf: {np.isinf(controls).any()}")
    
    # Angle of attack analysis
    theta = states[:, 2]
    alpha = theta - np.arctan2(vz, vx)
    
    print("\n🔍 Angle of Attack:")
    print(f"   Alpha range: [{np.degrees(alpha.min()):.2f}°, {np.degrees(alpha.max()):.2f}°]")
    extreme_alpha = (np.abs(alpha) > np.radians(30))
    if extreme_alpha.any():
        print(f"   ⚠️ {extreme_alpha.sum()} stages have |alpha| > 30°")
    
    # Reynolds number analysis  
    print("\n🔍 Reynolds Number Estimates:")
    rho = 1.225
    mu = 1.789e-5
    chord = 0.15  # Approximate
    Re = rho * v_mag * chord / mu
    print(f"   Re range: [{Re.min():.0f}, {Re.max():.0f}]")
    low_re = Re < 500
    if low_re.any():
        print(f"   ⚠️ {low_re.sum()} stages have Re < 500 (very low)")
    
    # Control bounds
    print("\n🔍 Control (Elevator) Analysis:")
    elevator = controls[:, 0]
    print(f"   Elevator range: [{np.degrees(elevator.min()):.2f}°, {np.degrees(elevator.max()):.2f}°]")
    
    # Look specifically at stage 25 (where variable 155 might be vz)
    print("\n🎯 Stage 25 Detailed Analysis (alternate interpretation of var 155):")
    stage_25 = states[25, :]
    print(f"   x={stage_25[0]:.4f}, z={stage_25[1]:.4f}")
    print(f"   theta={np.degrees(stage_25[2]):.2f}°, phi={np.degrees(stage_25[3]):.2f}°")
    print(f"   vx={stage_25[4]:.4f}, vz={stage_25[5]:.4f}")
    print(f"   |v|={np.sqrt(stage_25[4]**2 + stage_25[5]**2):.4f}")
    print(f"   alpha={np.degrees(stage_25[2] - np.arctan2(stage_25[5], stage_25[4])):.2f}°")
    
    return {
        'variable_155_stage': stage,
        'variable_155_component': state_names[component],
        'variable_155_value': states[stage, component],
        'min_velocity': v_mag.min(),
        'stages_below_vfloor': near_zero.sum(),
        'contains_nan': np.isnan(states).any() or np.isnan(controls).any()
    }


def compare_2d_vs_3d_initial_points():
    """Compare initial guess between 2D and 3D if available."""
    # This would require instrumenting go_safe_pdp.py to save initial guess
    pass


if __name__ == "__main__":
    print("="*80)
    print("INITIAL POINT ANALYSIS - WHERE DOES NaN COME FROM?")
    print("="*80)
    print("\nNote: We need to instrument go_safe_pdp.py to save the initial guess")
    print("      that IPOPT receives. For now, we can analyze the structure.")
    
    # Create a synthetic initial point based on config
    print("\n📝 Synthesizing initial guess from config...")
    n_stages = 111
    
    # From test.yaml init_state_ranges - take midpoints
    x_init = -10.0  # midpoint of [-12, -8]
    z_init = 0.75   # midpoint of [-1, 2.5]
    theta_init = 0.0
    phi_init = 0.0
    vx_init = 7.0   # midpoint of [3.5, 10.5]
    vz_init = 0.0   # midpoint of [-4, 4]
    
    # Create initial state trajectory (straight line guess)
    states = np.zeros((n_stages + 1, 6))
    for k in range(n_stages + 1):
        states[k, :] = [x_init, z_init, theta_init, phi_init, vx_init, vz_init]
    
    # Initial controls (zero elevator)
    controls = np.zeros((n_stages, 1))
    
    print(f"   Initial state: x={x_init}, z={z_init}, vx={vx_init}, vz={vz_init}")
    
    # Analyze this initial point
    results = find_problematic_values(states, controls)
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    if results['min_velocity'] > 0.1:
        print("✅ Velocities in initial guess are above floor (>0.1 m/s)")
        print("   NaN is NOT from initial velocity singularity")
        print("\n   🔍 NaN likely comes from:")
        print("   1. Chebyshev polynomial evaluation at the initial state")
        print("   2. Gradient computation through Chebyshev basis")
        print("   3. Constraint formulation involving cross-stage dependencies")
    else:
        print("⚠️ Initial guess has velocities below floor!")
        print(f"   Minimum velocity: {results['min_velocity']:.6f} m/s")
        print(f"   Stages affected: {results['stages_below_vfloor']}")
