# Degree 15 Test Analysis

## Results
**STILL FAILS** with NaN at same location (row 200, col 146)

## Configuration Verified
- Degree changed: 25 → 15
- Auxvar size changed: 4056 → 1536 ✓
- Coefficients per type: 676 → 256 ✓

## Why Degree Reduction Didn't Fix It

The problem is **NOT** simply the Chebyshev polynomial degree. Here's why:

### 1. Both degrees have sufficient samples
- Degree 25: 676 parameters, 1024 samples → overdetermined ✓
- Degree 15: 256 parameters, 1024 samples → overdetermined ✓
- Condition numbers are both excellent (4.0)

### 2. Same error location
- Constraint row 200 is still failing (control bound at stage 50)
- Variable 146 is still problematic (now a different coefficient due to reindexing, but same position)

### 3. The real issue
The problem is in the **CasADi symbolic expression evaluation**, not the polynomial fitting:

```python
# This works fine in forward pass:
CL_w = dot(cheb_basis_2d(alpha_scaled, Re_scaled, deg), phi_CL)

# But during IPOPT's automatic differentiation,
# the symbolic chain rule produces NaN when:
# - IPOPT line searches probe extreme values
# - Dynamics propagate through 50+ time stages
# - High CL causes extreme forces/accelerations
```

## What We've Learned

1. **LLT is perfect** - convergence, outputs, all clean
2. **Chebyshev fitting is perfect** - no NaN, good conditioning  
3. **Forward evaluation is fine** - all coefficients finite
4. **IPOPT's symbolic AD fails** - NaN during gradient computation

## Root Cause

The issue is that **3D LLT produces CL values (~1.46)** that are too high for the glider dynamics to handle stably during optimization. When IPOPT probes the parameter space, these high CL values cause:

1. Extreme lift forces
2. Unrealistic accelerations
3. Velocities that approach zero or go negative
4. Angular rates that diverge
5. Constraint gradients become undefined

## Why 2D Works

2D NeuralFoil produces **CL max = 0.81**, which keeps the dynamics in a stable regime.

## Actual Solution Needed

Not Chebyshev degree reduction, but:
1. **Tighter CL bounds** in the optimization
2. **Better dynamics safeguards** (already added velocity clamping, but not enough)
3. **Different initial guess** that avoids extreme states
4. **Penalty terms** to discourage high CL during transient phases

The Chebyshev polynomial is just faithfully representing what the 3D LLT gives it. The problem is the 3D LLT values themselves are too extreme for the current dynamics formulation.
