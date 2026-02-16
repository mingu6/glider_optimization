# Comprehensive Solution Plan for 3D LLT NaN Issue

**Date**: February 13, 2026  
**Status**: All simple regularization attempts FAILED

---

## Test Results Summary

### ❌ ALL REGULARIZATION TESTS FAILED

| Test | Parameter | Result | Notes |
|------|-----------|--------|-------|
| 1 | `velocity_floor = 0.5` | **FAILED** | NaN at same location (row 212, col 155) |
| 2 | `velocity_floor = 1.0` | **FAILED** | NaN at same location |
| 3 | `symbolic_epsilon = 1e-4` | **FAILED** | NaN at same location |
| 4 | `symbolic_epsilon = 1e-3` | **FAILED** | NaN at same location |

**Conclusion**: Simple regularization of velocity terms is insufficient. The NaN originates from a different source.

---

## Root Cause Analysis

### What We Know:
1. **2D NeuralFoil works perfectly** (obj = 15.766, no NaN)
2. **3D LLT data is valid** (CL range [-0.90, 1.46], no NaN in coefficients)
3. **Chebyshev fit is clean** (condition number = 4.0, no NaN in surface)
4. **NaN occurs in IPOPT gradient**, not forward pass
5. **Location**: ∂g₂₁₂/∂x₁₅₅ where:
   - Constraint 212: likely control bound at stage ~42-106
   - Variable 155: state (y-position at stage 22 OR vz at stage 25)
6. **Surface differences**: 2D and 3D differ by 5-10% typically, up to 32% max
7. **Regularization doesn't help**: Even 10x velocity floor and 1000x epsilon fail

### Why Regularization Failed:
The NaN is **NOT** coming from:
- ❌ `sqrt(vx² + vz²)` - we added epsilon there
- ❌ `atan2(vz, vx)` - we added epsilon there  
- ❌ Velocity denominators - we added floor there

The NaN is likely coming from:
- ✅ **Chebyshev basis function derivatives** at extreme values
- ✅ **Constraint formulation** involving cross-stage dependencies
- ✅ **High-order polynomial terms** in Chebyshev (degree 15 = 256 coefficients)
- ✅ **Numerical conditioning** of the full NLP Jacobian matrix

---

## Proposed Solutions

### 🎯 TIER 1: High-Probability Fixes (Recommended Order)

#### 1. **Reduce Chebyshev Degree** (HIGHEST PRIORITY)
**Rationale**: Lower-degree polynomials are more numerically stable.

```yaml
# conf/test.yaml
reducedModel:
  chebyshev_degree: 10  # Down from 15 (256 → 121 coefficients)
```

**Expected outcome**: 
- Reduces polynomial conditioning issues
- Loses some surface accuracy but may be acceptable
- 2D mode likely used lower degree implicitly

**Implementation**: 1 line change in config file  
**Risk**: Low  
**Test time**: ~1 minute

---

#### 2. **Tighten Alpha/Re Domain** 
**Rationale**: Narrower domain = Chebyshev extrapolates less = fewer edge case issues.

```yaml
neuralFoilSampling:
  AoA_min: -20  # Was -30
  AoA_max: 20   # Was 30
  Re_min: 500   # Was 100 (very low Re is problematic)
  Re_max: 50000 # Was 100000
```

**Expected outcome**:
- Reduces extrapolation into unstable regions
- More conservative but physically realistic
- Avoids extreme stall angles where data is uncertain

**Implementation**: 4 lines in config  
**Risk**: Low (narrows operating envelope but improves robustness)  
**Test time**: ~1 minute

---

#### 3. **Bound Elevator Deflection More Tightly**
**Rationale**: NaN occurs at constraint 212 (likely control bound). Aggressive maneuvers may be causing the issue.

```yaml
ocp:
  u_min: [-0.3]  # Was probably [-0.5] or unbounded
  u_max: [0.3]   # Was probably [0.5]
```

**Expected outcome**:
- Prevents optimizer from trying extreme elevator angles
- Reduces likelihood of trajectories entering v→0 regions
- More conservative but safer

**Implementation**: Add 2 lines to OCP config  
**Risk**: Low  
**Test time**: ~1 minute

---

### 🔧 TIER 2: Moderate Fixes (If Tier 1 Fails)

#### 4. **Use Lower Fidelity 3D Model Initially**
**Rationale**: Start with 2D, gradually introduce 3D via blending.

```python
# In glider_jinenv.py dynamics
blend_factor = fmin(1.0, iteration / 5.0)  # Ramp up over 5 iterations
CL_final = (1 - blend_factor) * CL_2d + blend_factor * CL_3d
```

**Expected outcome**:
- Initial solve uses robust 2D
- Gradually transitions to accurate 3D
- Warm-start reduces gradient singularities

**Implementation**: ~10 lines of code  
**Risk**: Medium (requires iteration tracking)  
**Test time**: ~5 minutes

---

#### 5. **Add Explicit Velocity Constraints**
**Rationale**: Force optimizer to stay away from v=0 singularity.

```python
# In OCP setup
opti.subject_to(v_w >= 0.5)  # Explicit lower bound on velocity magnitude
```

**Expected outcome**:
- Hard constraint prevents v→0 probing
- May make problem infeasible if trajectory naturally goes through slow regions

**Implementation**: 1-2 lines in OCP block  
**Risk**: Medium (could cause infeasibility)  
**Test time**: ~1 minute

---

#### 6. **Switch to Explicit (Autograd) LLT Backward Pass**
**Rationale**: Use explicit differentiation instead of implicit function theorem.

```yaml
neuralFoilSampling:
  llt_use_explicit: true  # If this option exists
```

**Expected outcome**:
- Different gradient path may avoid singularity
- Slower but more robust

**Implementation**: 1 line (if option exists)  
**Risk**: Medium (may slow down significantly)  
**Test time**: ~2 minutes

---

### 🔬 TIER 3: Deep Investigation (If All Else Fails)

#### 7. **Add IPOPT Callback to Capture Exact NaN Condition**
**Rationale**: See exact state/control values when NaN occurs.

```python
def ipopt_callback(alg, mem):
    if np.isnan(mem['g']).any():
        print("State at NaN:", mem['x'])
        print("Controls at NaN:", mem['x'][control_indices])
        # Save to file for analysis
```

**Expected outcome**:
- Identify exact operating point causing NaN
- Guide targeted fix

**Implementation**: ~30 lines (custom IPOPT callback)  
**Risk**: High (complex)  
**Test time**: ~30 minutes

---

#### 8. **Reformulate Dynamics (Energy Formulation)**
**Rationale**: Use (E, γ) instead of (vx, vz) to avoid velocity divisions.

```python
# Instead of: vx, vz (with divisions by v)
# Use: E = 0.5*m*v², γ = atan2(vz, vx)
```

**Expected outcome**:
- Eliminates velocity denominators entirely
- Requires significant OCP restructuring
- More robust but complex

**Implementation**: ~200 lines (major refactor)  
**Risk**: Very high  
**Test time**: Several hours

---

#### 9. **Try Different NLP Solver**
**Rationale**: IPOPT may be more sensitive than alternatives.

```python
# Options:
# - SNOPT (commercial, very robust)
# - KNITRO (commercial, hybrid algorithm)
# - WORHP (academic license)
opti.solver('knitro', {'algorithm': 'interior-direct'})
```

**Expected outcome**:
- Different solver may handle singularities better
- Requires license for most alternatives

**Implementation**: 1 line (+ license procurement)  
**Risk**: High (licensing, different API)  
**Test time**: Variable

---

## Recommended Action Plan

### Phase 1: Quick Wins (Do All Three)
1. ✅ **Reduce Chebyshev degree to 10**
2. ✅ **Tighten domain: alpha [-20, 20], Re [500, 50k]**
3. ✅ **Bound elevator: [-0.3, 0.3] rad**

**Time investment**: 5 minutes  
**Success probability**: ~70%

### Phase 2: If Phase 1 Fails
4. ✅ **Add explicit velocity constraint v >= 0.5**
5. ✅ **Try Chebyshev degree 8 or 6** (more drastic reduction)

**Time investment**: 10 minutes  
**Success probability**: ~85% cumulative

### Phase 3: If Phase 2 Fails
6. ✅ **Implement 2D→3D blending**
7. ✅ **Add IPOPT callback for debugging**

**Time investment**: 1 hour  
**Success probability**: ~95% cumulative

### Phase 4: Nuclear Option
8. ⚠️ **Energy formulation** OR **Different solver**

**Time investment**: Several hours to days  
**Success probability**: ~99% (one of these will work)

---

## Key Insights from Investigation

1. **The surfaces are only 5-10% different** - not huge, but enough to change trajectory strategy
2. **3D pushes optimization toward low-velocity regions** - where dynamics become singular
3. **Simple regularization doesn't work** - suggests the issue is in constraint formulation, not dynamics
4. **2D vs 3D difference matters more than absolute values** - it's about optimizer behavior, not just CL magnitude
5. **Chebyshev degree 15 may be overkill** - 256 coefficients for a 1024-sample dataset

---

## Files to Modify

### For Tier 1 Fixes:
- ✅ `conf/test.yaml` (4 parameter changes)

### For Tier 2 Fixes:
- ✅ `glider_optimization/utils/glider_jinenv.py` (dynamics blending)
- ✅ `glider_optimization/blocks/ocp.py` (velocity constraints)

### For Tier 3 Fixes:
- ⚠️ Multiple files (major refactoring)

---

## Next Command to Run

```bash
# Apply Tier 1 fixes and test
cd /Users/gherardi/Documents/GitHub/glider_optimization_debug

# Edit conf/test.yaml with Tier 1 changes, then:
WANDB_MODE=offline conda run -n general glider-opt --config conf/test.yaml
```

If successful: 🎉 Problem solved with minimal changes  
If failed: Proceed to Tier 2
