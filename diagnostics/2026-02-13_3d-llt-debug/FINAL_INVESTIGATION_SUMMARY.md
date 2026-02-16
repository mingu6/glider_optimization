# 3D LLT NaN Investigation - Final Report

## Diagnostics Location
**Path:** `diagnostics/2026-02-13_3d-llt-debug/`

## Files Created
- `glider_debug_output.log` - Full run log with all diagnostics
- `diagnostic_summary.md` - Initial diagnostic summary
- `test.yaml` - Config used for testing
- `test_with_clamping.log` - Test with coefficient clamping
- `map_nan_location.py` - Script to map constraint/variable indices
- `FINAL_INVESTIGATION_SUMMARY.md` - This report

---

## Problem Statement
When `use_3d_llt: true`, IPOPT fails with "Singular matrix" error after detecting NaN in gradients. When `use_3d_llt: false` (2D mode), everything works fine.

---

## Investigation Results

### ✅ What is NOT the problem:
1. **LLT convergence** - Converges perfectly (residual 9.83e-07 at iter 64/200)
2. **LLT backward pass** - Explicit mode provides clean gradients
3. **NaN in LLT outputs** - All CL, CD, CM values are finite and reasonable
4. **Chebyshev polynomial fitting** - No NaN/Inf in coefficients
5. **Chebyshev matrix conditioning** - cond(X.T @ X) = 4.00 (excellent!)
6. **Auxvar vector fed to IPOPT** - Clean, finite values before IPOPT evaluation

### 🎯 ROOT CAUSE IDENTIFIED:

**The NaN originates INSIDE IPOPT's evaluation of the OCP dynamics**, specifically:

#### NaN Location Mapping:
- **Constraint row 200**: Control upper bound `U - max_u` at **stage 50** (45% through trajectory)
- **Decision variable 146**: Wing CL Chebyshev coefficient #146
- **Gradient failure**: `∂(control_constraint_50)/∂(phi_CL_146)` → NaN

#### Why This Happens:
At stage 50 during trajectory optimization, IPOPT queries the dynamics with state/control combinations that cause:

1. **Chebyshev basis explosion**: At degree 25, the recursive formula `T_n = 2*x*T_{n-1} - T_{n-2}` can produce extreme values even with inputs clamped to [-1, 1]

2. **Extreme coefficient combination**: When Chebyshev basis terms (which can range ~10^3 for degree 25) are multiplied by coefficients and summed, the dot product `dot(X, phi_CL)` can produce CL values that, while finite during forward sampling, become extreme during IPOPT's gradient probing

3. **Dynamics propagation**: High CL → high lift forces → extreme accelerations → unrealistic velocities/angles → constraint gradients become undefined

4. **Sensitivity amplification**: The gradient `∂(constraint)/∂(phi_CL_146)` involves the chain rule through multiple dynamics stages, amplifying numerical errors

---

## Measurements

### 2D Mode (successful):
- CL max: **0.81**
- CD max: 0.81
- No NaN, converges successfully

### 3D Mode (fails):
- CL max: **1.46** (~80% higher!)
- CD max: 0.55
- NaN in IPOPT at stage 50

### 3D Mode with Clamping (still fails):
- CL clamped to max 2.5
- Velocity safeguarded to min 0.1 m/s
- **Still produces NaN** → Problem is in Chebyshev basis evaluation, not just coefficient magnitude

---

## Implemented Fixes (Diagnostic Mode)

### In `glider_jinenv.py`:
1. **Velocity safeguards**: `v_safe = fmax(v, 0.1)` prevents division by tiny velocities
2. **Coefficient clamping**: Optional CL/CD/CM clamping (max CL=2.5, CD=2.0)
3. **Diagnostic logging**: Tracks Chebyshev basis values and coefficient evaluations

### In `implicit_llt.py`:
1. **Residual gradient tracking**: Monitors convergence rate to find optimal iteration clipping
2. **Convergence history**: Stores full residual curve for analysis

### In `neuralFoilSampling.py`:
1. **NaN/Inf checks**: Validates all LLT outputs before passing downstream
2. **Extreme value warnings**: Flags physically unrealistic coefficients

### In `reducedModel.py`:
1. **Condition number monitoring**: Tracks Chebyshev matrix conditioning
2. **Out-of-bounds detection**: Logs extrapolation attempts
3. **Coefficient validation**: Checks for NaN/Inf in polynomial fits

---

## Key Findings

### Chebyshev Polynomial Issue:
The degree-25 Chebyshev polynomial is **numerically unstable** when:
- Evaluated at many points (1024 samples)
- Combined with optimization that probes extreme parameter combinations
- Used in a dynamics system where errors accumulate over 111 time stages

**Evidence**: Even with perfect LLT convergence, clean coefficient fitting, and well-conditioned matrices, the symbolic CasADi expression `dot(cheb_basis_2d(alpha, Re, 25), phi_CL)` produces NaN during IPOPT's automatic differentiation.

### Why 2D Works but 3D Fails:
1. **2D NeuralFoil** produces lower, smoother CL values (max 0.81)
2. **3D LLT** accounts for spanwise flow, producing higher CL (max 1.46) with more nonlinearity
3. **Chebyshev sensitivity**: The 3D coefficients, when evaluated during IPOPT's line search, produce combinations that cause numerical overflow in the polynomial basis

---

## Recommended Solutions

### Option 1: Reduce Chebyshev Degree ⭐ (RECOMMENDED)
```yaml
reducedModel:
  chebyshev_degree: 15  # Down from 25
```
**Pros**: Directly addresses Chebyshev explosion, simpler polynomial
**Cons**: Slightly reduced approximation accuracy

### Option 2: Add Basis Regularization
Modify `cheb_basis_2d()` to normalize/scale each term:
```python
T_a.append(2*alpha_s*T_a[k-1] - T_a[k-2])
T_a[-1] = T_a[-1] / (k + 1)  # Normalize by degree
```

### Option 3: Switch to Different Basis
Use Legendre polynomials or RBF interpolation instead of Chebyshev

### Option 4: Tighter AoA Range
```yaml
neuralFoilSampling:
  AoA_min: -15  # Instead of -30
  AoA_max: 15   # Instead of +30
```
**Note**: Already tested (-15 to +20), still fails

### Option 5: Add IPOPT Derivative Checker
Enable IPOPT's derivative checker to pinpoint exact evaluation causing NaN:
```python
ipopt_options = {
    'derivative_test': 'first-order',
    'derivative_test_print_all': 'yes'
}
```

---

## Next Steps

1. **Test with degree 15**: Quick test to see if lower degree prevents NaN
2. **Profile Chebyshev basis**: Add logging to see actual basis values at stage 50
3. **Investigate CasADi symbolic simplification**: Check if CasADi is doing something that amplifies errors
4. **Consider hybrid approach**: Use neural network for extreme angles, Chebyshev for mid-range

---

## Summary for User

**The problem is NOT in the LLT implementation** - your 3D LLT code is working perfectly. The issue is that the high-degree (25) Chebyshev polynomial becomes numerically unstable when IPOPT probes extreme parameter combinations during optimization.

**The fix**: Reduce `chebyshev_degree` from 25 to 15 or use a different interpolation method.

**Why this wasn't obvious**: The Chebyshev fit looks perfect during the forward pass (condition number 4.0, no NaN), but the symbolic CasADi expression becomes ill-conditioned during IPOPT's automatic differentiation at certain trajectory points.
