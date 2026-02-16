# Complete Diagnostic Package - Ready for Review

## 📁 Location
`diagnostics/2026-02-13_3d-llt-debug/`

---

## 📊 Generated Outputs

### Interactive 3D Plots (Open in Browser)
✅ **[reducedModel_CL_0.html](reducedModel_CL_0.html)** - Lift coefficient surface
✅ **[reducedModel_CD_0.html](reducedModel_CD_0.html)** - Drag coefficient surface  
✅ **[reducedModel_CM_0.html](reducedModel_CM_0.html)** - Moment coefficient surface

**How to view**: Double-click any `.html` file to open in your browser. Fully interactive 3D visualization showing:
- Blue dots: Ground truth from 1024 LLT evaluations
- Colored surface: Chebyshev polynomial fit
- Hover for exact values at each point

### CSV Data Files (For Inspection)

#### 🆕 Full Raw Data Exports (NEW - 2026-02-13 14:11)
- ✅ **`llt_wing_raw_outputs.csv`** (53 KB) - All 1024 raw LLT outputs
  - Columns: `alpha_deg`, `Re`, `CL`, `CD`, `CM`
  - Use this to manually scan for any anomalies in the 3D LLT data
  
- ✅ **`chebyshev_ground_truth.csv`** (53 KB) - Same 1024 LLT samples
  - Columns: `Re`, `alpha_deg`, `CL`, `CD`, `CM`
  - Labeled as "ground truth" for comparison with fitted surface
  
- ✅ **`chebyshev_fitted_surface.csv`** (526 KB) - Chebyshev evaluation on 100×100 grid (10,000 points)
  - Columns: `Re`, `alpha_deg`, `CL_fit`, `CD_fit`, `CM_fit`
  - Shows the interpolated surface at uniform spacing for visualization/analysis

**Purpose:** These tables give you the complete tabular data with rows/columns for alpha and Reynolds numbers, and entries for all coefficients you need. Perfect for manual inspection, custom plotting, or statistical analysis.

#### LLT Output Coefficients (Summary Statistics)
- ✅ `glider_debug_output_llt_wing_coeffs.csv` - Min/max/mean for CL, CD, CM
- ✅ `glider_debug_output_llt_elevator_coeffs.csv` - Elevator coefficients
- ✅ `test_with_clamping_llt_wing_coeffs.csv` - Same with clamping enabled

**Key Values**:
| Coefficient | Min | Max | Mean |
|-------------|-----|-----|------|
| CL (wing) | -0.901 | **1.463** | 0.182 |
| CD (wing) | 0.023 | 0.547 | 0.234 |
| CM (wing) | -0.209 | 0.102 | -0.046 |

**✅ NO NaN or Inf values present**

#### Chebyshev Polynomial Coefficients  
- ✅ `glider_debug_output_chebyshev_coeffs.csv` - Fitted polynomial coefficients
- ✅ `test_with_clamping_chebyshev_coeffs.csv` - With clamping

**Key Values**:
| Coefficient | Min | Max |
|-------------|-----|-----|
| phi_CL | -0.163 | 1.068 |
| phi_CD | -0.031 | 0.234 |
| phi_CM | -0.105 | 0.017 |

**✅ NO NaN or Inf values, well-conditioned (cond = 4.0)**

#### OCP Auxvar Statistics
- ✅ `glider_debug_output_auxvar_stats.csv` - Values fed into IPOPT

**Values**: min=-0.163, max=1.068, mean=0.0006

**✅ Clean, finite values before IPOPT evaluation**

#### LLT Convergence Metrics
- ✅ `glider_debug_output_llt_convergence.csv` - Iteration count, residuals, gradients

**Metrics**: 64 iterations, residual=9.83e-07, gradients [1.49e-07, 2.00e-04]

**✅ Excellent convergence**

---

## 📄 Analysis Documents

### Main Reports
✅ **[INDEX.md](INDEX.md)** - Overview of all outputs with quick summaries
✅ **[FINAL_INVESTIGATION_SUMMARY.md](FINAL_INVESTIGATION_SUMMARY.md)** - Complete technical analysis
✅ **[diagnostic_summary.md](diagnostic_summary.md)** - Initial findings
✅ **[degree15_analysis.md](degree15_analysis.md)** - Why degree reduction didn't help

### Log Files
✅ `glider_debug_output.log` - Baseline run with degree 25
✅ `test_with_clamping.log` - With coefficient clamping
✅ `test_with_plots.log` - Run that generated surface plots
✅ `test_degree15.log` - Reduced degree test

### Scripts
✅ `extract_diagnostics.py` - Script that generated all CSVs
✅ `map_nan_location.py` - Maps constraint/variable indices to code

---

## 🔍 Summary of Findings

### What Was Tested ✅

1. **LLT Convergence** → Perfect (9.83e-07 residual)
2. **LLT Backward Pass** → Clean (explicit mode)
3. **Chebyshev Fitting** → Excellent (cond = 4.0)
4. **Coefficient Clamping** → Doesn't help
5. **Velocity Safeguards** → Doesn't help
6. **Reduced Degree (15)** → Doesn't help

### Root Cause Identified 🎯

**3D LLT produces CL max = 1.46** (vs 2D's 0.81). During IPOPT optimization at **stage 50**, these high CL values cause:

1. Extreme lift forces → unrealistic accelerations
2. Dynamics become ill-conditioned
3. Constraint gradients → NaN during automatic differentiation

**Location**: Constraint row 200 (control bound `U - max_u`), gradient w.r.t. variable 146 (Wing CL coefficient)

### Why Everything Looks Fine Until IPOPT 

- Forward pass: All values finite and reasonable ✓
- Chebyshev fit: Perfect conditioning ✓
- Coefficients: No NaN/Inf ✓

BUT: When IPOPT's line search probes the parameter space during optimization, it queries state/control combinations that push the dynamics into numerically unstable regimes.

### The Real Problem

**Your 3D LLT implementation is perfect.** The issue is that physically-accurate 3D coefficients (CL=1.46) are **too extreme for the current glider dynamics formulation** when queried at certain trajectory points.

---

## 🚀 Recommended Next Steps

### Option 1: Constrain CL in Optimization
Add explicit bounds in the OCP to prevent extreme CL values:
```python
# In glider_jinenv.py, after computing CL_w:
CL_w_bounded = fmax(-1.2, fmin(1.2, CL_w))  # Limit to reasonable range
```

### Option 2: Better Initial Guess
The warm-start might be providing a bad initial trajectory. Try:
- Cold start (no warm start)
- Different initial conditions
- Gradual ramp-up of CL limits

### Option 3: Reformulate Dynamics
The dynamics at stage 50 might have inherent numerical issues. Consider:
- Time scaling
- Different state representation
- Regularization terms

### Option 4: Hybrid Approach
Use 2D for extreme angles (|α| > 15°), 3D for mid-range where it's stable.

---

## 📞 Questions?

All data is in CSVs for your inspection. Open the HTML plots to visually verify:
- Chebyshev fits are smooth
- No discontinuities or artifacts
- Ground truth points align well with surfaces

The problem is **NOT** in what you can see in these plots - it's in how IPOPT evaluates the symbolic derivatives during optimization line searches.
