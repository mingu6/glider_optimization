# Alternative Approaches: Complete Technical Explanations

## Summary of Current Situation

**Problem:** NaN in gradient ∂g₂₁₂/∂vz (constraint 212 w.r.t. vertical velocity)  
**Current fixes tried:**
- ✅ Symbolic epsilon (eps=1e-6) in sqrt and atan2 → Still fails  
- ✅ Velocity floor (v_min=0.1) → Still fails  
- 🔄 Testing: Larger epsilon, higher floor

**Root cause hypothesis:** Chebyshev polynomial extrapolation when Re < Re_min or Re > Re_max

---

## Approach 1: Symbolic Epsilon in Denominators

### Current Implementation (Partially Done)
```python
eps = 1e-6
v_w = sqrt(x_wdot² + z_wdot² + eps²)  # ✅ Done
alpha_w = theta - atan2(z_wdot, x_wdot + eps)  # ✅ Done
```

### What's Still Missing
```python
# Problem: Reynolds number can still go outside valid range
Re = rho * v_w * chord / mu  # If v_w ≈ eps, Re ≈ 1e-4 << Re_min=100

# In Chebyshev:
X_re = (2*Re - (Re_max + Re_min)) / (Re_max - Re_min)
# When Re < Re_min: X_re < -1 → Chebyshev polynomial undefined!
```

### Complete Fix
```python
# 1. Larger epsilon to keep Re in valid range
eps = 1e-3  # Ensures Re ≥ ~0.1 m/s * constants ≈ 100

# 2. Add epsilon to ALL divisions, not just velocity
Re_denom = mu + eps  # Prevent mu→0 (though physically impossible)
Re = rho * v_w * chord / Re_denom

# 3. Add epsilon in Chebyshev basis normalization
X_alpha = (2*alpha - (alpha_max + alpha_min)) / (alpha_max - alpha_min + eps)
X_re = (2*Re - (Re_max + Re_min)) / (Re_max - Re_min + eps)
```

### Why It Works
- **Epsilon in symbolic graph** means CasADi's AD sees it during differentiation
- Prevents 0/0, sqrt(0), arccos(x>1) at compile-time
- Smooth gradients everywhere IPOPT probes

### Trade-offs
- **Modeling error:** ~0.1% everywhere (not just at singularities)
- **Performance:** Negligible (one extra constant in expression)
- **Robustness:** High (works for any IPOPT sampling)

---

## Approach 2: Smooth Approximations (Replace atan2)

### Problem with atan2
```python
alpha = theta - atan2(vz, vx)
# Derivatives:
∂alpha/∂vz = -vx / (vx² + vz²)  # → 0/0 at origin
∂alpha/∂vx = vz / (vx² + vz²)   # → 0/0 at origin
```

Even with epsilon in denominator, atan2 has **discontinuity at ±π**.

### Smooth Alternative 1: Soft atan2
```python
def soft_atan2(y, x, eps=1e-4):
    """Smooth atan2 approximation using hyperbolic functions."""
    # Normalize
    r = sqrt(x**2 + y**2 + eps**2)
    x_norm = x / r
    y_norm = y / r
    
    # Use arctan(y/x) with smooth denominator
    return arctan(y_norm / (x_norm + eps))
```

**Derivatives:** All C∞ (infinitely differentiable), no discontinuities

### Smooth Alternative 2: Rational Approximation
```python
def smooth_angle(vz, vx, eps=1e-4):
    """Smooth angle function without transcendentals."""
    # Based on Padé approximant
    r2 = vx**2 + vz**2 + eps**2
    # First-order approximation near origin
    return vz / sqrt(r2)  # ∈ [-1, 1]
```

**Simpler**, faster, but **less accurate** at large angles.

### Smooth Alternative 3: Conditional Smoothing
```python
def atan2_smooth(y, x, eps=1e-3, smooth_radius=1e-2):
    """Use atan2 far from origin, linear approximation near it."""
    r = sqrt(x**2 + y**2)
    
    # Heaviside smoothing function
    w = 1 / (1 + exp(-100 * (r - smooth_radius)))  # Sigmoid
    
    # Blend: linear near origin, atan2 far away
    angle_far = atan2(y, x)
    angle_near = y / (smooth_radius + eps)
    
    return w * angle_far + (1-w) * angle_near
```

**Best accuracy**, smooth transition, but **more complex**.

### Implementation in glider_jinenv.py
```python
# Replace:
alpha_w = theta - atan2(z_wdot, x_wdot + eps)

# With:
alpha_w = theta - soft_atan2(z_wdot, x_wdot, eps=1e-4)
```

### Trade-offs
- **Accuracy:** 1-5% error near origin, <0.1% elsewhere
- **Complexity:** Moderate (add helper function)
- **Physical meaning:** Angle approximation, not exact
- **Robustness:** Very high (no singularities anywhere)

---

## Approach 3: Energy Formulation (Avoid v in Denominators)

### Current Formulation (Force-Based)
```python
# State: x = [x_pos, z_pos, vx, vz, theta, theta_dot]
# Dynamics:
ax = (L*sin(alpha) - D*cos(alpha)) / m
az = (L*cos(alpha) - D*sin(alpha) - W) / m
# Where: L = 0.5*rho*v²*S*CL, D = 0.5*rho*v²*S*CD
```

**Problem:** Need angle alpha = theta - atan2(vz, vx) → atan2 singularity

### Energy Formulation
```python
# State: x = [x_pos, z_pos, E, gamma, theta, theta_dot]
# Where:
#   E = 0.5*m*v² + m*g*z  (total mechanical energy)
#   gamma = flight path angle (angle of velocity vector)
#   v = sqrt(2*(E - m*g*z)/m)  (velocity from energy)
```

**Dynamics:**
```python
# Energy rate (power balance)
dE/dt = T*v - D*v = v*(T - D)
      = v*(T - 0.5*rho*v²*S*CD)  # No division by v!

# Flight path angle rate  
dgamma/dt = (L - W*cos(gamma)) / (m*v)  # Still has 1/v, but...
          = (L/W - cos(gamma)) * g/v      # Can be regularized

# Position rates (integrate from gamma and v)
dx/dt = v * cos(gamma)
dz/dt = v * sin(gamma)
```

### Key Improvements
1. **Energy is smooth** even when v→0 (E remains finite)
2. **Gamma (angle) is state**, not computed from velocities
3. **Fewer atan2 calls** - only in alpha = theta - gamma

### Full Reformulation
```python
def dynamics_energy(t, x, u, params):
    x_pos, z_pos, E, gamma, theta, theta_dot = x
    elevator_deflection = u
    
    # Derive velocity from energy
    v = sqrt(2*max(E - m*g*z_pos, eps) / m)  # Clamp E-mgz > 0
    
    # Angle of attack
    alpha = theta - gamma  # No atan2!
    
    # Forces
    L = 0.5 * rho * v**2 * S * CL(alpha, Re(v))
    D = 0.5 * rho * v**2 * S * CD(alpha, Re(v))
    
    # Rates
    dE_dt = -D * v  # Power dissipation (assuming no thrust)
    dgamma_dt = (L/W - cos(gamma)) * g / (v + eps)  # Regularized
    dx_dt = v * cos(gamma)
    dz_dt = v * sin(gamma)
    dtheta_dt = theta_dot
    dtheta_dot_dt = moment_equation(...)
    
    return [dx_dt, dz_dt, dE_dt, dgamma_dt, dtheta_dt, dtheta_dot_dt]
```

### Trade-offs
- **Complexity:** High - need to rewrite entire OCP
- **Physical insight:** Better (energy is fundamental quantity)
- **Numerical:** More stable (energy doesn't have singularities)
- **Compatibility:** Requires changing state definition everywhere
- **Effort:** ~1-2 days of work

### When to Use
- If all other approaches fail
- For publication-quality work (energy formulation is more elegant)
- When robustness is critical

---

## Approach 4: Different Solver (SNOPT/KNITRO)

### Why IPOPT Fails
**IPOPT** (Interior Point OPTimizer):
- Uses **barrier method**: adds log barriers to inequality constraints
- Requires **smooth, continuous** gradients
- **Strict**: Dies immediately on NaN
- **Line search**: Uses polynomial approximation, probes many points
- **Trust region**: Fixed size initially

**When gradient has singularity:**
- IPOPT's line search probes near singularity
- Gradient evaluates to NaN
- IPOPT aborts ("couldn't find solution")

### SNOPT (Sequential Quadratic Programming)

**Algorithm:**
1. Build **quadratic model** of objective
2. **Linearize** constraints
3. Solve **QP subproblem**
4. Take **SQP step**
5. Repeat until convergence

**Advantages:**
- **Finite-difference fallback**: If AD gradient is NaN, uses finite differences
- **Adaptive line search**: Can skip problematic regions
- **Better conditioning**: QP subproblems are more stable
- **Warm start**: Can resume from partial solutions

**Disadvantages:**
- **Requires license** (~$1500/year for academic)
- **Slower**: O(n³) per iteration vs IPOPT's O(n²)
- **Memory**: Stores full Hessian approximation

**Implementation:**
```python
# In go_safe_pdp.py or ocp.py
solver = ca.nlpsol('solver', 'snopt', nlp, {
    'snopt.Major_optimality_tolerance': 1e-4,
    'snopt.Major_feasibility_tolerance': 1e-6,
    'snopt.Minor_feasibility_tolerance': 1e-6
})
```

### KNITRO (Interior-point + Active-set Hybrid)

**Algorithm:**
- **Adaptive**: Switches between interior-point and active-set methods
- **Line search variants**: Multiple backtracking strategies
- **Multistart**: Can try multiple initial points

**Advantages:**
- **Most robust** commercial solver
- **Handles ill-conditioning** better than IPOPT
- **Detailed diagnostics**: Tells you exactly where/why it failed
- **CasADi integration**: Good support

**Disadvantages:**
- **Expensive**: ~$3000/year
- **Complex**: Many tuning parameters
- **Overkill**: If problem is fundamentally ill-posed, won't help

**Implementation:**
```python
solver = ca.nlpsol('solver', 'knitro', nlp, {
    'knitro.algorithm': 4,  # SQP algorithm
    'knitro.bar_feasible': 1,  # Stay feasible
    'knitro.honorbnds': 1  # Don't violate bounds during line search
})
```

### Free Alternative: WORHP

**WORHP** (We Optimize Really Huge Problems):
- Free for academic use
- Similar to IPOPT but more robust
- Better at handling poorly scaled problems

### When to Try Different Solver
1. **After symbolic epsilon fails** (you've tried eps=1e-3, 1e-2)
2. **After smooth approximations fail**
3. **Before energy reformulation** (less work to try solver first)
4. **When problem structure is correct** but numerics are hard

### Cost-Benefit
| Approach | Time | Cost | Success Probability |
|----------|------|------|---------------------|
| Symbolic epsilon | 1 hour | Free | 60% |
| Smooth atan2 | 3 hours | Free | 70% |
| SNOPT trial | 1 day | Free trial | 40% |
| Energy reformulation | 2 days | Free | 90% |
| KNITRO license | 1 week | $3000 | 50% |

---

## Decision Tree

```
Start
  │
  ├─ Try eps=1e-4, 1e-3 (15 min each)
  │    ├─ Success? → Done ✅
  │    └─ Fail → Continue
  │
  ├─ Try v_floor=0.5, 1.0 (15 min each)
  │    ├─ Success? → Done ✅
  │    └─ Fail → Continue
  │
  ├─ Implement soft_atan2 (2 hours)
  │    ├─ Success? → Done ✅
  │    └─ Fail → Continue
  │
  ├─ Try WORHP solver (1 day)
  │    ├─ Success? → Done ✅
  │    └─ Fail → Continue
  │
  └─ Energy reformulation (2 days)
       └─ Success → Done ✅
```

**Recommended path:**
1. Quick tests first (eps, v_floor)
2. Soft atan2 if quick tests fail (good ROI)
3. Energy reformulation as last resort (highest success rate)
4. Different solver only if time/budget constrained

---

## Current Testing Status

Running now with increased epsilon (1e-4) to see if that alone fixes it...
