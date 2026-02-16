# Deep Investigation: Gradient NaN Root Cause

## Current Status

**Symbolic epsilon 1e-6 implemented** in dynamics but **still fails** at same location:
- Row 212 (constraint)
- Col 155 (vertical velocity vz)
- NaN in ∂g₂₁₂/∂vz

## What We Know

### 1. The Symbolic Expressions

In `glider_jinenv.py`, the dynamics include:

```python
# Wing velocity (NOW with symbolic epsilon)
eps = 1e-6
v_w = sqrt(x_wdot² + z_wdot² + eps²)  # Prevents ∂v/∂z singularity at origin

# Angle of attack (NOW with symbolic epsilon)  
alpha_w = theta - atan2(z_wdot, x_wdot + eps)  # Prevents ∂alpha/∂x, ∂alpha/∂z singularities
```

**Gradient of velocity:**
```
∂v_w/∂z_wdot = z_wdot / sqrt(x_wdot² + z_wdot² + eps²)
```
✅ **Well-defined everywhere** (denominator ≥ eps > 0)

**Gradient of angle of attack:**
```
∂alpha_w/∂z_wdot = -(x_wdot + eps) / ((x_wdot + eps)² + z_wdot²)
∂alpha_w/∂x_wdot = z_wdot / ((x_wdot + eps)² + z_wdot²)
```
✅ **Well-defined everywhere** (denominator ≥ eps² > 0)

### 2. Why eps=1e-6 Still Fails

**Theory:** The NaN doesn't come from these direct derivatives, but from:

**A) Reynolds Number Gradients**
```python
Re = rho * v_w * chord / mu
# When v_w is very small (near eps), Re ≈ rho * eps * chord / mu ≈ 1e-6 * ... ≈ 1e-4

# In Chebyshev evaluation:
X_re = (2*Re - (Re_max + Re_min)) / (Re_max - Re_min)
# With Re_min=100, Re_max=100000:
X_re = (2*Re - 100100) / 99900

# If Re → 0 (velocity near eps):
X_re ≈ -100100/99900 ≈ -1.002  # OUTSIDE [-1, 1]!
```

**Chebyshev polynomials are only valid for x ∈ [-1, 1]**. When Re goes below Re_min or above Re_max, the polynomial **extrapolates** and can produce:
- Extremely large values
- Oscillations
- NaN in higher-order terms

**B) Chebyshev Polynomial High-Order Terms**

With degree 15, the highest term is T₁₅(x):
```
T₁₅(x) = cos(15 * arccos(x))  for x ∈ [-1, 1]
```

When x slightly exceeds [-1, 1], arccos(x) becomes complex, causing NaN!

**C) Compound Gradients Through Chebyshev**

The constraint likely involves:
```python
# Simplified
g[212] = something with CL, CD from Chebyshev(alpha_w, Re)
# Gradient chain:
∂g/∂vz = ∂g/∂CL * ∂CL/∂Re * ∂Re/∂v_w * ∂v_w/∂vz
       + ∂g/∂CL * ∂CL/∂alpha * ∂alpha/∂vz
```

Even though ∂v_w/∂vz and ∂alpha/∂vz are fine, the **Chebyshev gradients** ∂CL/∂Re and ∂CL/∂alpha explode when Re or alpha exit the training domain!

## Testing Plan

### Test 1: Larger Symbolic Epsilon (eps = 1e-4)
**Hypothesis:** eps=1e-6 is too small, velocities still get close enough to cause Re < Re_min

**Implementation:**
```python
eps = 1e-4  # 100x larger
```

**Expected:** Keeps Re ≥ ~0.01, well within [100, 100000] range  
**Risk:** More modeling error (v_min effectively becomes ~1 cm/s)

### Test 2: Higher Velocity Floor (v_min = 0.5 or 1.0 m/s)
**Hypothesis:** Even with symbolic epsilon, IPOPT probes very low velocities during line search

**Implementation:**
```python
v_min = 0.5  # or 1.0
v_w_safe = fmax(v_w, v_min)
```

**Expected:** Forces velocities to stay well above minimum  
**Risk:** Large modeling error if true optimal trajectory has v < 0.5 m/s

### Test 3: Clamp Reynolds Number to Valid Range
**Hypothesis:** Need to explicitly prevent Re from exiting Chebyshev domain

**Implementation:**
```python
Re = rho * v_w * chord / mu
Re_safe = fmax(Re_min + eps, fmin(Re_max - eps, Re))  # Clamp to [100.0001, 99999.9999]
```

**Expected:** Chebyshev never extrapolates  
**Risk:** Discontinuous gradient at boundaries

### Test 4: Smooth Reynolds Clamping with Sigmoid
**Hypothesis:** Need smooth transition, not hard clamp

**Implementation:**
```python
# Sigmoid soft clamp
def soft_clamp(x, x_min, x_max, sharpness=10):
    # Smoothly maps R → [x_min, x_max]
    t = (x - x_min) / (x_max - x_min)  # Map to [0, ∞)
    t_smooth = 1 / (1 + exp(-sharpness * (t - 0.5)))  # Sigmoid
    return x_min + t_smooth * (x_max - x_min)

Re_safe = soft_clamp(Re, Re_min, Re_max)
```

**Expected:** Smooth gradients everywhere, Re always in valid range  
**Risk:** More complex, slower evaluation

## Investigation Tools

### A) Log Actual Values When NaN Occurs

Add to OCP right before solver call:

```python
# Create callback to capture state when NaN detected
def ipopt_callback(alg_mod, iter_count, obj_value, inf_pr, inf_du, 
                   mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
    if iter_count == 0:  # Log initial point
        x0 = alg_mod.x_init
        # Extract states at all stages
        for k in range(n_stages):
            vx_k = x0[k*7 + 3]  # Assuming interleaved layout
            vz_k = x0[k*7 + 5]
            v_k = np.sqrt(vx_k**2 + vz_k**2)
            if v_k < 0.2:
                print(f"⚠️ Stage {k}: v={v_k:.6f}, vx={vx_k:.6f}, vz={vz_k:.6f}")
    return True

opts = {
    'iteration_callback': ipopt_callback,
    ...
}
```

### B) Visualize Trajectory Velocities

After failed optimization:
```python
# Extract trajectory from failed solution
X_failed = sol['x'][:n_states*(n_stages+1)].reshape(n_stages+1, n_states)
vx_traj = X_failed[:, 3]
vz_traj = X_failed[:, 5]
v_traj = np.sqrt(vx_traj**2 + vz_traj**2)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(v_traj, 'o-')
plt.axhline(v_min, color='r', linestyle='--', label=f'v_min={v_min}')
plt.xlabel('Stage')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Trajectory')
plt.legend()

plt.subplot(122)
Re_traj = rho * v_traj * chord / mu
plt.plot(Re_traj, 'o-')
plt.axhline(Re_min, color='r', linestyle='--', label='Re_min')
plt.axhline(Re_max, color='g', linestyle='--', label='Re_max')
plt.xlabel('Stage')
plt.ylabel('Reynolds Number')
plt.title('Reynolds Number Trajectory')
plt.legend()
plt.tight_layout()
plt.savefig('diagnostics/failed_trajectory_analysis.png')
```

## Quick Manual Tests

Running now:
1. ✅ Baseline (eps=1e-6, v_min=0.1) → FAILED at row 212, col 155
2. 🔄 Test eps=1e-4
3. 🔄 Test v_min=0.5
4. 🔄 Test eps=1e-3 (even larger)

## Alternative Explanations

If none of the above work, the NaN might come from:

### 1. Constraint Formulation Issue
The constraint g[212] itself might have a problematic structure unrelated to velocity derivatives.

**Check:** What IS constraint 212? Need to inspect `env.path_inequ` definition.

### 2. Auxiliary Variable Dependency
The constraint might depend on auxiliary variables (Chebyshev coefficients) in a numerically unstable way.

**Check:** Does ∂g/∂auxvar contain Chebyshev polynomial evaluations at extreme points?

### 3. CasADi Expression Graph Issue
The symbolic expression tree might have accumulated numerical issues from multiple operations.

**Check:** Simplify the expression graph, use `simplify()` on CasADi functions.

### 4. IPOPT Line Search Pathology
IPOPT might be probing invalid regions during its line search that have nothing to do with the actual trajectory.

**Check:** Reduce `alpha_init` or change line search strategy in IPOPT options.
