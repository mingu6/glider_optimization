"""
Map CasADi constraint/variable indices to meaningful names.

NaN Location from test with CL=0.8:
- grad_f_x: row 166
- jac_g_x: row 212, col 155
- g: row 212

This script determines what constraint 212 and variable 155 represent.
"""

import yaml
from pathlib import Path

# Load config to get OCP parameters
config_path = Path("conf/test.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

n_stages = 111  # Hardcoded in ocp.py
dt = 0.027  # Hardcoded in glider_jinenv.py

# System dimensions
n_states = 6  # [x, y, z, vx, vy, vz]
n_controls = 1  # [elevator_deflection]

# Chebyshev coefficients
degree = cfg.get("reducedModel", {}).get("chebyshev_degree", 15)
n_cheb_coeffs_per_surface = (degree + 1) ** 2  # 16^2 = 256

# Decision variables structure:
# - States: (n_stages + 1) * n_states = 112 * 6 = 672
# - Controls: n_stages * n_controls = 111 * 1 = 111
# Total decision vars: 672 + 111 = 783

# Auxiliary variables (parameters, not optimized):
# - Wing: 3 surfaces * 256 coeffs = 768
# - Elevator: 3 surfaces * 256 coeffs = 768 (in 3D mode)
# Total auxvars: 1536 (3D) or 768 (2D)

print("=" * 80)
print("DECISION VARIABLE MAPPING")
print("=" * 80)

var_idx = 155
print(f"\n🎯 Variable {var_idx} (where NaN occurs in jac_g_x)")
print("-" * 80)

# Variables are laid out as: [states_0, controls_0, states_1, controls_1, ..., states_111]
# OR: [all_states, all_controls]
# Need to check go_safe_pdp.py for exact layout

# Common layouts:
# Layout 1: [X_0, U_0, X_1, U_1, ..., X_N]
# Layout 2: [X_0, X_1, ..., X_N, U_0, U_1, ..., U_{N-1}]

# For Layout 1 (interleaved):
vars_per_stage = n_states + n_controls  # 7
if var_idx < (n_stages + 1) * n_states + n_stages * n_controls:
    # Interleaved layout
    stage_local = var_idx // vars_per_stage
    offset_in_stage = var_idx % vars_per_stage
    
    if offset_in_stage < n_states:
        var_type = "STATE"
        state_names = ["x", "y", "z", "vx", "vy", "vz"]
        var_name = state_names[offset_in_stage]
        print(f"Type: {var_type}")
        print(f"Stage: {stage_local}/{n_stages}")
        print(f"Time: {stage_local * dt:.3f} s")
        print(f"Component: {var_name} (state #{offset_in_stage})")
    else:
        var_type = "CONTROL"
        control_offset = offset_in_stage - n_states
        control_names = ["elevator_deflection"]
        var_name = control_names[control_offset] if control_offset < len(control_names) else f"u_{control_offset}"
        print(f"Type: {var_type}")
        print(f"Stage: {stage_local}/{n_stages}")
        print(f"Time: {stage_local * dt:.3f} s")
        print(f"Component: {var_name}")

# For Layout 2 (all states, then all controls):
print(f"\nAlternate interpretation (blocked layout):")
total_states = (n_stages + 1) * n_states  # 672
if var_idx < total_states:
    stage = var_idx // n_states
    state_offset = var_idx % n_states
    state_names = ["x", "y", "z", "vx", "vy", "vz"]
    print(f"Type: STATE")
    print(f"Stage: {stage}/{n_stages}")
    print(f"Time: {stage * dt:.3f} s")
    print(f"Component: {state_names[state_offset]} (state #{state_offset})")
else:
    control_idx = var_idx - total_states
    stage = control_idx // n_controls
    control_offset = control_idx % n_controls
    print(f"Type: CONTROL")
    print(f"Stage: {stage}/{n_stages}")
    print(f"Time: {stage * dt:.3f} s")
    print(f"Component: elevator_deflection (control #{control_offset})")

print("\n" + "=" * 80)
print("CONSTRAINT MAPPING")
print("=" * 80)

constr_idx = 212
print(f"\n🎯 Constraint {constr_idx} (where NaN occurs in jac_g_x and g)")
print("-" * 80)

# Constraint structure (from initConstraints and path_inequ):
# Per stage (111 stages):
#   - Dynamics: n_states equations (implicit Euler or similar)
#   - Path constraints: elevator deflection bounds, velocity/altitude limits
#   - Control bounds: typically 2 per control (lower, upper)
# Terminal constraints: final state
# Initial constraints: initial state

# Typical constraint count per stage:
# - Control bounds: 2 * n_controls = 2
# - State bounds: some subset (e.g., altitude > 0, velocity limits)
# Let's estimate ~4-6 constraints per stage

# If ~5 constraints/stage * 111 stages = 555 constraints
# Plus initial (6) + terminal (6) = ~567 total

# Row 212 out of ~567 means:
stage_estimate = constr_idx // 5  # Rough estimate
print(f"Estimated stage: ~{stage_estimate}/{n_stages}")
print(f"Estimated time: ~{stage_estimate * dt:.3f} s")

# More precise: need to look at env.path_inequ structure
print(f"\nLikely constraint types:")
print(f"  1. Control bound: U_min ≤ elevator_deflection ≤ U_max")
print(f"  2. State path constraint: e.g., altitude > 0, velocity limits")
print(f"  3. Dynamics residual (if treated as inequality): x_{{k+1}} = x_{{k}} + dt*f(x,u)")

# From the error message "row 200" (unclamped) vs "row 212" (clamped):
# This shift suggests the constraint structure changed slightly
# Most likely: control bound at a specific stage

# Row 200 / 2 constraints per control per stage = stage 100
# Row 212 / 2 = stage 106
print(f"\n📊 If constraint 212 is control bound (2 per stage):")
print(f"   Stage: {constr_idx // 2} (if only control bounds)")
print(f"   Type: {'Upper bound' if constr_idx % 2 == 1 else 'Lower bound'}")

print(f"\n📊 If constraint 212 includes path constraints:")
# Typical: [lower_bound, upper_bound] per control + other path constraints
# Need to check glider_jinenv.py initConstraints()

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print(f"""
The NaN occurs in the GRADIENT ∂g_{constr_idx}/∂x_{var_idx}, not in the values themselves.

This means:
- The constraint value g[{constr_idx}] might be finite
- The state/control value x[{var_idx}] might be finite
- But the symbolic derivative is undefined (division by zero, sqrt of negative, etc.)

The most likely culprits:
1. Division by velocity in dynamics: ∂/∂v of (something/v) when v→0
2. Square root in velocity: ∂/∂vx of sqrt(vx² + vz²) when both are small
3. Arctan in angle of attack: ∂/∂v of atan2(vz, vx) when denominator is tiny
4. Chebyshev basis evaluation: ∂/∂coeff of polynomial at extreme alpha/Re

To find the exact symbolic expression:
- Instrument CasADi's NLP solver callback
- Print state/control values at iteration when NaN occurs
- Examine the constraint equation for stage {stage_estimate}
""")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("""
1. Add IPOPT iteration callback to capture state/control values when NaN appears
2. Check glider_jinenv.py for constraint equations around stage {}/{}
3. Examine symbolic expression for constraint {} 
4. Look for operations like:
   - v_w or v_e appearing in denominators
   - atan2(vz, vx) derivatives
   - sqrt(vx² + vz²) derivatives when velocities are small
   - Chebyshev polynomial high-order terms
5. Test with higher velocity floor (v_min = 0.5 or 1.0 m/s)
""".format(stage_estimate, n_stages, constr_idx))
