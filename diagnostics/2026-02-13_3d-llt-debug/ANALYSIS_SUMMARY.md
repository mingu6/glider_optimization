# 3D vs 2D Comparison & Velocity Floor Testing

## Results Summary (2026-02-13 14:30)

### Task 1: 2D Diagnostics ✅ COMPLETE

Successfully ran `use_3d_llt: false` mode with same diagnostic exports:

**Files Created:**
- `neuralfoil_2d_raw_outputs.csv` (53 KB) - All 1024 2D NeuralFoil predictions
- `chebyshev_ground_truth.csv` (53 KB) - Ground truth for Chebyshev fit
- `chebyshev_fitted_surface.csv` (526 KB) - 100×100 grid evaluation
- 3 HTML plots: CL, CD, CM surfaces

**Result:** ✅ **2D mode SUCCEEDS** - No NaN, optimization converges
- Objective: 15.766
- All coefficients clean and finite

---

### Task 2: Remove CL Clamping ✅ COMPLETE

**Change:** Restored `CL_max = 2.5` (from 0.8) in `glider_jinenv.py` line 223

**Justification:** CL=0.8 clamping didn't prevent NaN, only shifted location (row 200→212, col 146→155). This rules out high CL values as the root cause.

---

### Task 3: Higher Velocity Floor 🔄 IN PROGRESS

**Implementation:** Added configurable velocity floor in `glider_jinenv.py`:
```python
v_min = getattr(self, '_velocity_floor', 0.1)  # Default 0.1 m/s
v_w_safe = fmax(v_w, v_min)
v_e_safe = fmax(v_e, v_min)
```

**Test Plan:**
1. ✅ Current: v_min = 0.1 m/s (FAILS with NaN)
2. 🔄 Test: v_min = 0.5 m/s
3. 🔄 Test: v_min = 1.0 m/s

**Hypothesis:** Higher floor prevents division by near-zero velocity in symbolic gradients.

---

### Task 4: Map Variable/Constraint Indices ✅ COMPLETE

Created `map_variable_constraint.py` - Analysis results:

**Variable 155 (where ∂g/∂x = NaN):**
- **Most likely:** State #5 (vz = vertical velocity) at stage 25
- **Time:** ~0.675 seconds into trajectory
- **Alternative:** State #1 (y = lateral position) at stage 22

**Constraint 212 (where g and ∂g/∂x have NaN):**
- **Estimated stage:** ~42-106 (depending on constraint layout)
- **Time:** ~1.1-2.9 seconds
- **Type:** Likely control bound or path constraint

**KEY FINDING:** 
The NaN is in the **GRADIENT** ∂g₂₁₂/∂x₁₅₅, not the values themselves!

This means:
- Constraint value g[212] might be finite
- Variable value x[155] might be finite
- But the symbolic derivative involves:
  - Division by velocity (when v→0)
  - sqrt(vx² + vz²) derivative (undefined when both small)
  - atan2(vz, vx) derivative (unstable near origin)

---

### Task 5: IPOPT Callback 🔄 IN PROGRESS

Need to add iteration callback to capture:
- Exact state/control values when NaN first appears
- Which IPOPT iteration triggers the NaN
- Current line search parameters

---

## Key Insights

### 1. **Problem is in Symbolic Gradient, Not Values**

The NaN occurs during CasADi's automatic differentiation when IPOPT queries gradients. The actual trajectory values might be fine, but the symbolic expression for ∂g/∂x becomes undefined.

### 2. **2D Works, 3D Fails**

- 2D NeuralFoil: CL max = 0.81 → ✅ Success
- 3D LLT (unclamped): CL max = 1.46 → ❌ NaN at row 200
- 3D LLT (CL≤0.8): CL max ≤ 0.8 → ❌ NaN at row 212 (shifted but still fails)

**Conclusion:** Not purely a CL magnitude issue.

### 3. **Likely Root Cause: Velocity Derivatives**

The gradient involves velocity (vz or v_w) in stage 25-42. Suspect operations:
```python
# In dynamics:
alpha_w = theta - atan2(z_wdot, x_wdot)  # ∂/∂vz is unstable when vx,vz→0
v_w = sqrt(x_wdot² + z_wdot²)            # ∂/∂vz is undefined at origin
Re = rho * v_w * chord / mu              # ∂/∂v_w

# In Chebyshev evaluation:
X_re = (2*Re - (Re_max + Re_min)) / (Re_max - Re_min)  # If Re varies wildly
```

### 4. **Why 3D Fails But 2D Works**

**Hypothesis:** 3D LLT produces coefficient gradients that:
1. Create more extreme Reynolds number variations
2. Lead to trajectories with lower velocities
3. Push the optimization into regions where velocity→0
4. Trigger undefined symbolic gradients

---

## Dynamics Regularization Explanation

**Current safeguard:**
```python
v_w_safe = fmax(v_w, 0.1)  # Clip velocity to minimum 0.1 m/s
```

**Problem:** This only affects the **forward evaluation**, not the **symbolic gradient**!

When CasADi builds the symbolic expression graph:
```python
v_w = sqrt(x_wdot² + z_wdot²)
alpha = theta - atan2(z_wdot, x_wdot)
```

The gradient ∂alpha/∂z_wdot contains:
```
∂/∂z_wdot [atan2(z_wdot, x_wdot)] = -x_wdot / (x_wdot² + z_wdot²)
```

If IPOPT probes a point where both x_wdot and z_wdot are near zero (even if later clipped), the symbolic gradient evaluates to 0/0 = NaN.

**Solution with higher floor:**
- Larger v_min (e.g., 0.5-1.0 m/s) makes the "safe zone" bigger
- Reduces probability IPOPT samples unsafe regions
- BUT: Introduces modeling error when true velocity < v_min

---

## Comparison: 2D vs 3D Coefficients

Need to analyze the CSV files to compare:

| Metric | 2D NeuralFoil | 3D LLT | Difference |
|--------|---------------|--------|------------|
| CL min | TBD | -0.901 | TBD |
| CL max | TBD | 1.463 | TBD |
| CD min | TBD | 0.023 | TBD |
| CD max | TBD | 0.547 | TBD |
| Success | ✅ Yes | ❌ No | NaN at ~1s |

---

## Next Actions

1. **Test v_min = 0.5 m/s**:
   ```python
   # In glider_jinenv.py __init__:
   self._velocity_floor = 0.5
   ```

2. **Test v_min = 1.0 m/s**:
   ```python
   self._velocity_floor = 1.0
   ```

3. **Add IPOPT callback** to capture NaN-triggering state

4. **Compare CSV files** to quantify 2D vs 3D differences

5. **If velocity floor doesn't help**: Consider alternative approaches:
   - Use sqrt(v² + epsilon) instead of sqrt(v²) + clipping
   - Replace atan2 with smoothed approximation
   - Add small epsilon to denominators symbolically, not just in evaluation
   - Use different NLP solver (e.g., SNOPT, SQP-based)

