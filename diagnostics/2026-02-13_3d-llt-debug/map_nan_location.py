#!/usr/bin/env python3
"""Map OCP constraint and variable indices to understand NaN source."""

horizon = 111
n_states = 8  # [x, z, theta, phi, xdot, zdot, thetadot, t]
n_controls = 1  # [phidot]

# Path constraints: 4 per stage
# 1. U - max_u (control upper bound)
# 2. -U - max_u (control lower bound)
# 3. X[3] - max_phi (elevator angle upper bound)
# 4. -X[3] + min_phi (elevator angle lower bound)

total_path_constraints = horizon * 4
print(f"Horizon: {horizon} stages")
print(f"Path constraints per stage: 4")
print(f"Total path constraints: {total_path_constraints}")

print(f"\n{'='*70}")
print("CONSTRAINT ROW 200:")
print(f"{'='*70}")
stage = 200 // 4
local_idx = 200 % 4
constraint_names = ["U - max_u", "-U - max_u", "phi - max_phi", "-phi + min_phi"]
print(f"  Stage: {stage} (out of {horizon})")
print(f"  Local index: {local_idx}")
print(f"  Type: {constraint_names[local_idx]}")
print(f"\n  → This is the ELEVATOR ANGLE LOWER BOUND constraint at stage {stage}")

# Decision variables: [auxvar, states, controls]
deg = 25
n_cheb_params = (deg + 1) ** 2  # 676 parameters per coefficient type
n_wing_params = 3 * n_cheb_params  # CL, CD, CM for wing = 2028
n_elev_params = 3 * n_cheb_params  # CL, CD, CM for elevator = 2028
n_auxvar = n_wing_params + n_elev_params  # 4056 total

print(f"\n{'='*70}")
print("DECISION VARIABLE 146:")
print(f"{'='*70}")
print(f"  Chebyshev params (auxvar): {n_auxvar}")
print(f"    Wing params: {n_wing_params}")
print(f"    Elevator params: {n_elev_params}")
print(f"  States: {horizon * n_states}")
print(f"  Controls: {horizon * n_controls}")

if 146 < n_auxvar:
    print(f"\n  → Variable 146 is a CHEBYSHEV COEFFICIENT")
    print(f"     Coefficient index: {146}")
    if 146 < n_cheb_params:
        print(f"     Type: Wing CL coefficient {146}")
    elif 146 < 2 * n_cheb_params:
        print(f"     Type: Wing CD coefficient {146 - n_cheb_params}")
    elif 146 < 3 * n_cheb_params:
        print(f"     Type: Wing CM coefficient {146 - 2 * n_cheb_params}")
else:
    idx_in_traj = 146 - n_auxvar
    if idx_in_traj < horizon * n_states:
        stage_idx = idx_in_traj // n_states
        state_idx = idx_in_traj % n_states
        state_names = ["x", "z", "theta", "phi", "xdot", "zdot", "thetadot", "t"]
        print(f"\n  → Variable 146 is STATE: {state_names[state_idx]}")
        print(f"     At stage: {stage_idx}")
    else:
        control_idx = idx_in_traj - horizon * n_states
        print(f"\n  → Variable 146 is CONTROL (phidot)")
        print(f"     At stage: {control_idx}")

print(f"\n{'='*70}")
print("ANALYSIS:")
print(f"{'='*70}")
print("NaN appears in:")
print("  1. grad_f_x at row 157 → Likely Wing CD coefficient 157")
print("  2. jac_g_x at (row 200, col 146) → Elevator angle constraint w.r.t. Wing CL coeff 146")
print("  3. constraint g at row 200 → Elevator angle lower bound at stage 50")
print("")
print("ROOT CAUSE HYPOTHESIS:")
print("  The Chebyshev polynomial evaluation at stage 50 produces extreme values")
print("  that propagate through the dynamics, causing the elevator angle constraint")
print("  to become ill-defined (possibly due to extreme lift forces causing")
print("  unrealistic accelerations or velocities).")
print("")
print("  Stage 50 is approximately 45% through the trajectory, where the glider")
print("  is likely transitioning between flight regimes.")
