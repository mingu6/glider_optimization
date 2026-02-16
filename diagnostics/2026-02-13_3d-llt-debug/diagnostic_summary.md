# 3D LLT Diagnostic Results - 2026-02-13

## Test Configuration
- AoA range: -30° to +30°
- LLT: max_iter=200, backward_mode=explicit
- Samples: 1024
- Chebyshev degree: 25
- max_outer_iters: 1 (first iteration only)

## Diagnostic Results

### ✅ 1. LLT Output Check: CLEAN
**Wing Coefficients (3D LLT):**
- CL: min=-0.901385, max=1.463340, mean=0.182259
- CD: min=0.023226, max=0.547452, mean=0.233687
- CM: min=-0.209154, max=0.101864, mean=-0.045831

**Elevator Coefficients:**
- CL_e: min=-1.095999, max=1.095999, mean=0.000000
- CD_e: min=0.018246, max=0.518630, mean=0.231390
- CM_e: min=-0.156889, max=0.156889, mean=0.000000

**Status:** No NaN/Inf detected. Values physically reasonable.

### ✅ 2. LLT Convergence: EXCELLENT
- Converged at iteration 64/200
- Final residual: 9.83e-07 (below tol=1e-06)
- Residual gradients: 
  - Recent (iter 60-64): 1.49e-07
  - Mid-range (iter 15-20): 2.00e-04

### ✅ 3. Chebyshev Polynomial Coefficients: CLEAN
- phi_CL: [-0.1632, 1.0674]
- phi_CD: [-0.0314, 0.2337]
- phi_CM: [-0.1050, 0.0168]

**Status:** No NaN/Inf detected in coefficients.

### ✅ 4. Chebyshev Matrix Conditioning: EXCELLENT
- cond(X.T @ X) = 4.00
- **Well-conditioned!** No numerical issues in fitting.

### ✅ 5. OCP Auxvar Vector: CLEAN
- Shape: (4056, 1)
- Range: [-0.163197, 1.068424]
- Mean: 0.000598

**Status:** No NaN/Inf before feeding to IPOPT.

---

## 🔴 Problem Identified: NaN Originates in IPOPT

### Error Location:
1. **grad_f_x (row 157, col 0)**: Gradient of objective w.r.t. decision variable 157
2. **jac_g_x (row 200, col 146)**: Jacobian of constraint 200 w.r.t. variable 146
3. **constraint g (row 200, col 0)**: Constraint value itself is NaN

### Root Cause:
The NaN does **NOT** originate in:
- ❌ LLT fixed-point iteration
- ❌ LLT backward pass
- ❌ Chebyshev polynomial fitting
- ❌ Chebyshev matrix conditioning

The NaN originates **INSIDE IPOPT's evaluation** of the OCP dynamics/constraints.

### Hypothesis:
When IPOPT queries certain state/control combinations during optimization, the dynamics equations in `glider_jinenv.py` produce NaN. This likely happens when:
- High CL values (1.46) cause extreme lift forces
- Velocities approach zero or negative values
- Angles cause division by zero in aerodynamic calculations
- sqrt/log operations receive invalid inputs

---

## Next Steps:

1. **Map constraint index 200** to specific dynamics equation
2. **Add IPOPT callback logging** to capture exact state/control values
3. **Audit glider_jinenv.py** for numerical hazards:
   - Division by velocity
   - sqrt() of negative values
   - log() of non-positive values
4. **Test with CL clamping** (max CL = 1.0) to verify hypothesis

---

## Files:
- Full log: `glider_debug_output.log`
- Config: `../../conf/test.yaml`
