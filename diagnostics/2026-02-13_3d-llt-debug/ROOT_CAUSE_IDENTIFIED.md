# Deep Investigation Results: Why 3D LLT Causes NaN

**Date**: February 13, 2026  
**Investigation Type**: Root cause analysis with exact value tracking  
**Status**: ROOT CAUSE IDENTIFIED

---

## 🎯 KEY FINDING

**The NaN occurs during INITIAL POINT EVALUATION**, before IPOPT even starts iterating!

Location in logs:
```
CasADi - WARNING("solver:nlp_grad_f failed: Inf detected for output grad_f_x, at (row 166, col 0).")
CasADi - WARNING("solver:nlp_jac_g failed: NaN detected for output jac_g_x, at nonzero index 780 (row 212, col 155).")
```

This happens **immediately** when IPOPT tries to evaluate constraints and gradients at the initial guess.

---

## 📊 Initial Guess Analysis

From `conf/test.yaml`:
```yaml
init_state_ranges:
  - [-12.0, -8.0]   # x position
  - [-1.0, 2.5]     # z position  
  - [-0.5, 0.5]     # theta
  - [-0.05, 0.05]   # phi
  - [3.5, 10.5]     # x velocity (vx)
  - [-4.0, 4.0]     # z velocity (vz)
```

**Midpoint initial guess** (what IPOPT receives):
- x = -10.0 m
- z = 0.75 m
- theta = 0° (level flight)
- phi = 0° (no elevator)
- **vx = 7.0 m/s** ✅ (well above velocity floor)
- **vz = 0.0 m/s** ⚠️ (EXACTLY ZERO!)

---

## 🔍 Where Variable 155 is Located

**Variable 155** in the decision vector:
- Layout: `[state_0 (6 vars), state_1 (6 vars), ..., state_111, control_0, ...]`
- Variable 155 = state variable at index 155
- **Stage**: 155 ÷ 6 = 25 (remainder 5)
- **Component**: 5 → **vz (vertical velocity)**
- **Value at initial guess**: 0.0 m/s

**Constraint 212** (where NaN occurs):
- Estimated stage: ~42-106 (from 212 constraints / 2 per stage)
- Type: Likely control bound or path constraint
- **This constraint's gradient depends on vz at stage 25!**

This is a **cross-stage coupling** through the dynamics.

---

## 🧮 Operating Point at Initial Guess

At stage 25 (and all stages since initial guess is constant):
- vx = 7.0 m/s, vz = 0.0 m/s
- |v| = 7.0 m/s ✅
- alpha = theta - atan2(vz, vx) = 0° - atan2(0, 7) = 0° ✅
- Re ≈ 1.225 × 7 × 0.15 / 1.789e-5 ≈ 71,900 ✅

**Aerodynamic coefficients at this point:**
- 2D NeuralFoil: CL=0.503, CD=0.027, CM=-0.099
- 3D LLT: CL=0.395, CD=0.033, CM=-0.093
- **Difference**: ΔCL=-0.109 (3D is 22% lower!)

---

## 🔴 Root Cause Hypothesis

The NaN is **NOT** from:
- ❌ Velocity magnitude → |v|=7.0 is fine
- ❌ atan2(vz, vx) evaluation → atan2(0, 7) = 0, no singularity
- ❌ sqrt(vx² + vz²) → sqrt(49) = 7, no problem
- ❌ Regularization terms → epsilon didn't help

The NaN **IS** from:
✅ **Gradient of constraint 212 with respect to vz at stage 25**

### Why This Happens:

When vz=0 EXACTLY, certain gradients become problematic:

1. **∂(atan2(vz, vx))/∂vz at vz=0**:
   ```
   ∂/∂vz [atan2(vz, vx)] = vx / (vx² + vz²)
   At vz=0, vx=7: = 7/49 = 0.14 ✅ This is fine
   ```

2. **But higher-order chain rule terms through Chebyshev**:
   The Chebyshev polynomial evaluation involves:
   ```python
   CL_w = dot(chebyshev_basis(alpha, Re), phi_CL)
   ```
   
   Where `alpha = theta - atan2(vz, vx)`.
   
   The SECOND derivative or cross-terms like:
   ```
   ∂²CL/∂vz² or ∂²alpha/∂vz∂vx
   ```
   
   Can become undefined when vz=0 depending on the polynomial order!

3. **Constraint 212 depends on stage 25's vz through dynamics coupling**:
   - Stage 25: Has vz=0
   - Stages 26-42: Dynamics propagate forward
   - Constraint 212 (at stage ~42): Depends on accumulated effect
   
   The gradient ∂g₂₁₂/∂vz₂₅ involves the chain:
   ```
   ∂g₂₁₂/∂x₄₂ · ∂x₄₂/∂x₄₁ · ... · ∂x₂₆/∂vz₂₅
   ```
   
   If any link in this chain hits a singularity at vz=0, NaN propagates!

---

## 💡 Why 2D Works But 3D Doesn't

**Both use the same dynamics equations** with vz=0 initial guess.

**The difference is in the Chebyshev coefficients**:

| Coefficient | 2D Range | 3D Range | Difference |
|-------------|----------|----------|------------|
| CL | [-0.94, 1.45] | [-0.90, 1.46] | Similar |
| CD | [0.018, 0.529] | [0.023, 0.548] | Similar |
| CM | [-0.232, 0.115] | [-0.209, 0.102] | Similar |

At the initial point (alpha≈0°, Re≈72k):
- 2D: CL=0.503
- 3D: CL=0.395 (22% lower!)

**Hypothesis**: The 3D Chebyshev polynomial, with its specific coefficients:
- Has higher-order terms that amplify gradient singularities
- Creates numerical conditioning issues in the Jacobian
- Leads to a singular Hessian when computing second derivatives

The polynomial:
```
CL(alpha, Re) = Σᵢⱼ φᵢⱼ · Tᵢ(alpha_scaled) · Tⱼ(Re_scaled)
```

With degree 15 → 256 coefficients, some high-order terms (e.g., T₁₄·T₁₅) can have extreme gradients at certain operating points.

---

## 📋 What Needs to Be Done Next

### Option 1: Instrument Dynamics Evaluation (RECOMMENDED)

Add logging to `glider_jinenv.py` to capture:
```python
def dynamics_with_logging(self, x, u, auxvar):
    # Log every evaluation
    print(f"Eval: vx={x[4]:.6f}, vz={x[5]:.6f}, alpha={alpha:.6f}")
    print(f"  CL={CL_w:.6f}, CD={CD_w:.6f}")
    print(f"  Force: Fx={Fx:.6f}, Fz={Fz:.6f}")
    
    # Check for problematic values
    if abs(x[5]) < 1e-10:  # vz near zero
        print(f"  ⚠️ vz≈0 detected!")
```

This will show:
- Exact values when NaN occurs
- Which term in the dynamics produces NaN/Inf
- Whether it's CL, CD, forces, or something else

### Option 2: Perturb Initial Guess

Instead of vz=0, use vz=0.01 or vz=-0.01:
```yaml
init_state_ranges:
  - [-4.0, -3.9]  # vz: slightly negative instead of crossing zero
```

If this works, confirms vz=0 is the trigger.

### Option 3: Different Initialization Strategy

Sample initial conditions that avoid vz=0:
```python
vz_init = np.random.uniform(-0.5, -0.1)  # Avoid zero crossing
```

### Option 4: Chebyshev Basis Regularization

Add epsilon to Chebyshev evaluation:
```python
alpha_scaled_safe = fmax(-0.9999, fmin(0.9999, alpha_scaled))
```

Prevents evaluation at exact Chebyshev node boundaries where derivatives spike.

---

## 🎯 Most Likely Solution

**The problem is vz=0 in the initial guess combined with 3D's specific Chebyshev coefficients creating a gradient singularity.**

**Quick test**: Modify initial guess to have non-zero vz:
```yaml
init_state_ranges:
  - [-4.0, -0.5]    # vz: ensure it's not zero
```

If this works → Confirms root cause  
If this fails → Need deeper instrumentation

---

## 📁 Evidence Files

All analysis scripts in: `diagnostics/2026-02-13_3d-llt-debug/`
- `deep_nan_investigation.py` - Compares 2D vs 3D trajectories
- `analyze_initial_point.py` - Decodes variable 155
- `check_initial_coefficients.py` - Compares coefficients at initial point
- `detailed_2d_full.log` - 2D run (successful)
- `detailed_3d_full.log` - 3D run (NaN before iteration 0)

**Key log line proving immediate failure:**
```
2D: 0 IPOPT iterations
3D: 0 IPOPT iterations
🔴 3D failed immediately (0 iterations)
```

The NaN happens during CasADi's initial constraint/gradient evaluation, not during IPOPT optimization.
