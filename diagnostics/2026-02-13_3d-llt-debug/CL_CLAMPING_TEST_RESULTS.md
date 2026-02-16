# CL Clamping to 0.8 - Test Results

**Date:** 2026-02-13 14:11:28  
**Test:** Run 3D LLT with CL clamped to 0.8 (matching 2D NeuralFoil territory)  
**Hypothesis:** High CL values (1.46) from 3D LLT cause NaN. Clamping to 2D territory (0.8) should prevent this.

---

## ❌ Result: **STILL FAILS WITH NaN**

Even with CL clamped to 0.8 in the dynamics equations, IPOPT still encounters NaN during optimization.

---

## Test Configuration

### Changes Made:
1. **CL Clamping** (`glider_jinenv.py` line 223):
   ```python
   CL_max = 0.8  # Match 2D NeuralFoil territory (was 2.5)
   CL_w = fmax(-CL_max, fmin(CL_max, CL_w))
   CL_e = fmax(-CL_max, fmin(CL_max, CL_e))
   ```

2. **CSV Exports Added**:
   - `llt_wing_raw_outputs.csv` - All 1024 raw LLT outputs (alpha, Re, CL, CD, CM)
   - `chebyshev_ground_truth.csv` - Same 1024 LLT samples for reference
   - `chebyshev_fitted_surface.csv` - 100×100 grid (10,000 points) from Chebyshev polynomial evaluation

### LLT Configuration:
- Max iterations: 200
- Convergence tolerance: 1e-6
- Backward mode: implicit (IFT)
- Chebyshev degree: 25

---

## Failure Details

### CasADi Warnings:
```
CasADi - WARNING("solver:nlp_grad_f failed: NaN detected for output grad_f_x, at (row 166, col 0).")
CasADi - WARNING("solver:nlp_jac_g failed: NaN detected for output jac_g_x, at nonzero index 780 (row 212, col 155).")
CasADi - WARNING("solver:nlp_g failed: NaN detected for output g, at (row 212, col 0).")
```

### Location Change:
| Metric | Unclamped (CL≤2.5) | Clamped (CL≤0.8) | Change |
|--------|-------------------|------------------|--------|
| **grad_f_x NaN row** | 157 | 166 | +9 |
| **jac_g_x NaN row** | 200 | 212 | +12 |
| **jac_g_x NaN col** | 146 | 155 | +9 |

**Constraint row 212** = Different control bound than row 200  
**Variable col 155** = Different Chebyshev coefficient than col 146

---

## Key Finding: **Clamping Does NOT Prevent NaN**

### Implications:
1. **High CL values are NOT the sole cause** - NaN persists even with CL ≤ 0.8
2. **Problem is deeper** - Likely related to:
   - Dynamics equation formulation (force/moment calculations)
   - Velocity division by near-zero denominators
   - CasADi symbolic expression evaluation during line search
   - Constraint formulation sensitivity

3. **NaN location shifts** - Clamping changes where NaN appears, but doesn't eliminate it

---

## LLT Convergence (Still Perfect)

All 3 airfoil geometries converged successfully:
```
✓ LLT converged at iteration 106/200, residual=7.23e-07
✓ LLT converged at iteration 64/200, residual=9.83e-07
✓ LLT converged at iteration 56/200, residual=8.88e-07
```

**Chebyshev fit quality:**
- Condition number: 4.0 (excellent)
- CL coefficient range: [-0.163, 1.067] → **Max CL still exceeds 0.8!**
- CD coefficient range: [-0.031, 0.234]
- CM coefficient range: [-0.105, 0.017]

**⚠️ Important:** The Chebyshev coefficients (phi_CL) range up to 1.067, which means the fitted surface can still produce CL > 0.8 even though we clamp in the dynamics. The clamping happens *after* Chebyshev evaluation, so IPOPT's symbolic differentiation sees the unclamped expressions first.

---

## CSV Export Files

### 1. llt_wing_raw_outputs.csv (53 KB)
- **Rows:** 1,024 (32×32 Chebyshev nodes)
- **Columns:** alpha_deg, Re, CL, CD, CM
- **Purpose:** Raw 3D LLT outputs before Chebyshev fitting
- **Sample:**
  ```
  alpha_deg,Re,CL,CD,CM
  29.963863,99939.836,0.99469334,0.3595853,-0.15352322
  29.963863,99459.37,0.9926985,0.35945424,-0.15326181
  ```

### 2. chebyshev_ground_truth.csv (53 KB)
- **Rows:** 1,024
- **Columns:** Re, alpha_deg, CL, CD, CM
- **Purpose:** Same as #1, labeled as "ground truth" for comparison
- **Note:** Identical to llt_wing_raw_outputs.csv

### 3. chebyshev_fitted_surface.csv (526 KB)
- **Rows:** 10,000 (100×100 uniform grid)
- **Columns:** Re, alpha_deg, CL_fit, CD_fit, CM_fit
- **Purpose:** Chebyshev polynomial evaluation on dense grid for visualization
- **Sample:**
  ```
  Re,alpha_deg,CL_fit,CD_fit,CM_fit
  160.16696,-29.963863,-0.7216895,0.46592426,0.049647003
  1168.0425,-29.963863,-0.81349826,0.33775622,0.07886946
  ```

---

## Interactive Plots

Generated 3 HTML files with LLT points (blue dots) + Chebyshev surface:
- `reducedModel_CL_0.html` - CL surface
- `reducedModel_CD_0.html` - CD surface  
- `reducedModel_CM_0.html` - CM surface

**Confirmed:** In the plots, **blue dots = LLT outputs**, **colored surface = Chebyshev polynomial fit**

---

## Next Steps

Since CL clamping to 0.8 **did not solve the problem**, the issue must be elsewhere:

### Potential Root Causes:
1. **Velocity division issues** - Despite safeguards (v_min=0.1), symbolic expressions may still encounter numerical issues
2. **Force calculation sensitivity** - Lift/drag forces: `F = 0.5 * rho * v² * S * C`  
   - When v² varies, small changes in CL cause large force variations
   - CasADi's AD may produce undefined gradients at certain trajectory points

3. **Constraint formulation** - Control bounds interacting poorly with coefficient-dependent dynamics

4. **Chebyshev basis evaluation** - Despite low condition number, symbolic differentiation of polynomial basis may be unstable

### Recommended Approaches:

#### Option A: Alternative Coefficient Representation
- Use Legendre polynomials instead of Chebyshev
- Try B-splines or radial basis functions
- Test with simple bilinear interpolation

#### Option B: Constraint Reformulation  
- Replace hard constraints with soft penalties
- Add slack variables to control bounds
- Use trust-region instead of line search

#### Option C: Dynamics Regularization
- Add higher velocity floor (v_min = 1.0 m/s)
- Scale coefficients by confidence factor
- Smooth coefficient transitions with moving average

#### Option D: Hybrid 2D/3D Approach
- Use 2D for extreme angles (|alpha| > 20°)
- Use 3D only in mid-range where convergence is stable
- Blend coefficients in transition regions

#### Option E: Different Initial Guess
- Start from 2D-converged trajectory
- Gradually increase 3D contribution (continuation method)
- Warm-start with simplified dynamics

---

## Conclusion

**The NaN issue is NOT caused by high CL values alone.** Even with CL clamped to 2D NeuralFoil territory (≤0.8), IPOPT still encounters NaN during optimization, though at different locations (row 212 vs 200, col 155 vs 146).

The problem appears to be in the **interaction between the dynamics equations and IPOPT's symbolic automatic differentiation**, not in the LLT or Chebyshev fitting themselves (which are both working perfectly).

**All requested CSV exports are complete** and ready for manual inspection in:
- `diagnostics/2026-02-13_3d-llt-debug/llt_wing_raw_outputs.csv`
- `diagnostics/2026-02-13_3d-llt-debug/chebyshev_ground_truth.csv`
- `diagnostics/2026-02-13_3d-llt-debug/chebyshev_fitted_surface.csv`
