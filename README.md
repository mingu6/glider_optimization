# glider_optimization_lite

Gradient-based airfoil shape optimization for agile glider manoeuvres (perching, soft landing).  
The airfoil geometry is co-optimized with the flight trajectory end-to-end, differentiating through aerodynamics → reduced model → optimal control.

---

## Method overview

The optimizer is structured as a differentiable **bilevel program**: the outer level optimizes the airfoil shape; the inner level solves a constrained Optimal Control Problem (OCP) whose aerodynamic tables are rebuilt from the current shape at every outer iteration.

The full forward pass chains five blocks sequentially:

```
Airfoil  →  NeuralFoilSampling[3D]  →  ReducedModel  →  OCP  →  Evaluation
```

Gradients flow back through the chain via a custom block-wise backward pass.  No automatic differentiation is threaded through the OCP solver itself; instead, sensitivity information is extracted analytically via the Pontryagin Maximum Principle (PMP / IDOC identities on the auxiliary Constrained Optimal Control system).

### Airfoil parameterization

Both variants use the **Kulfan (CST) parameterization** — 8 upper + 8 lower B-spline weights, one leading-edge weight, one trailing-edge thickness scalar — giving a smooth, low-dimensional design space with built-in aerodynamic constraints (upper ≥ lower + gap, TE thickness clipped).  Parameters are optimized with Adam + exponential LR decay.

---

## 2D mode (`use_3d_llt: false`)

**Blocks:** `Airfoil` + `NeuralFoilSampling`

A **single root-section** airfoil is optimized.  Aerodynamic coefficients $(C_L, C_D, C_M)$ over the operating envelope are obtained by querying **NeuralFoil** (a neural-network surrogate of XFOIL) at a Chebyshev grid of $(\alpha, Re)$ sample points.

The 2D mode is intentionally simple: it treats the wing as spanwise-uniform and ignores 3D effects entirely.  It is best suited for rapid prototyping or cases where induced drag and spanwise load redistribution are not critical.

**Sampling:**  $\sqrt{N} \times \sqrt{N}$ Chebyshev nodes in $[\alpha_\text{min}, \alpha_\text{max}] \times [Re_\text{min}, Re_\text{max}]$.  A separate 20% random validation set monitors out-of-sample fit quality.  An augmented-Lagrangian penalty enforces a minimum mean NeuralFoil confidence and a minimum mean $C_L/C_D$.

---

## 3D mode (`use_3d_llt: true`)

**Blocks:** `Airfoil3D` + `NeuralFoilSampling3D`

The wing is treated as a **tapered, swept, dihedraled lifting surface** and aerodynamic coefficients are evaluated via a differentiable **Lifting-Line Theory (LLT)** solver coupled with NeuralFoil as the local 2D section model.

### Root + tip parameterization

Two independent Kulfan airfoils are optimized simultaneously — one at the root, one at the tip — with **linear spanwise interpolation** between them at each panel's $\eta = y / (b/2)$ station.  This allows the optimizer to exploit washout, camber taper, and thickness taper along the span.

### LLT solver (`utils/llt.py`)

The LLT is implemented as a **Picard fixed-point iteration** on the bound-circulation distribution $\Gamma^*$:

$$\Gamma^{(k+1)} = (1-\beta)\,\Gamma^{(k)} + \beta\,\Gamma^*({\alpha_\text{eff}^{(k)}})$$

where $\alpha_\text{eff}$ is the local effective angle of attack after subtracting the near-field downwash, and $\Gamma^*$ is the circulation implied by the local 2D $C_L$ from NeuralFoil.  The iteration terminates when $\|\Delta\Gamma\|_\infty < \text{tol}$ or a hard cap on iterations is reached.

Aerodynamic features that make the 3D model physically sound:

- **Sweep correction** — all local velocities, chords, and Reynolds numbers are projected onto the panel-normal direction via $\cos\Lambda_i$ (horizontal-plane sweep angle per panel), giving the classical Prandtl–Glauert-consistent section conditions.
- **Dihedral** — panel positions and control-point $z$-offsets are computed from the dihedral angle, correctly displacing the vortex filaments and control points out of the $y$–$z$ plane.
- **Twist** — per-panel geometric twist is added to the geometric angle of attack before downwash subtraction.
- **Induced drag (Trefftz plane)** — induced drag is integrated in the Trefftz far-field plane ($D_i = \rho V \Gamma w_\text{tr}$), consistent with the Flow5 methodology, giving more reliable drag polars than near-field pressure integration.
- **Profile drag** — sectional profile drag from NeuralFoil is integrated with the $\cos^2\Lambda$ sweep factor.
- **Pitching moment** — $C_M$ is integrated about the local quarter-chord with a $\cos^3\Lambda$ factor, then referenced to the global wing quarter-chord reference point.
- **Symmetry enforcement** — left/right panel circulations are averaged each Picard step, suppressing numerical asymmetry from floating-point accumulation.
- **Dynamic wing reference geometry** — when `dynamic_centroid: true`, the aerodynamic center and wing centroid are recomputed every iteration from the actual airfoil cross-section polygonal centroid (Kulfan → polygon → centroid), feeding the correct moment arms and reference area into the flight dynamics model.
- **Differentiability** — the entire LLT solve is wrapped in a `torch.autograd.Function` (`LLTImplicitFn`).  The forward pass runs the Picard loop in `no_grad` mode for efficiency; the backward pass applies the **implicit function theorem** on the fixed-point residual $F(\Gamma^*) = 0$ to compute exact gradients of the converged $\Gamma^*$ wrt airfoil parameters without unrolling the iteration.
- **Per-panel confidence** — a second NeuralFoil call on the converged $\alpha_\text{eff}$ and local $Re$ at each panel yields a spanwise confidence map; its mean is used in the augmented-Lagrangian constraint to keep the optimization inside the NeuralFoil training distribution.

### Unsteady extension *(planned)*

The current 3D LLT is quasi-static (steady attached flow at each OCP time step).  A first-order unsteady extension based on the **Wagner / indicial-response** formulation of **Sugar Gabor** is under development, combined with a first-order lag filter on the effective angle of attack to approximate delayed boundary-layer separation at high $\alpha$.  This will allow the optimizer to reason about dynamic stall onset during aggressive perching manoeuvres.

---

## Reduced aerodynamic model (`ReducedModel`)

The $(C_L, C_D, C_M)$ tables produced by NeuralFoilSampling[3D] are compressed into a **2D tensor-product Chebyshev polynomial**:

$$C(\alpha, Re) \approx \sum_{i,j=0}^{d} \phi_{ij}\,T_i(\bar\alpha)\,T_j(\overline{Re})$$

Coefficients $\phi$ are solved by ridge regression (L2-regularized least squares) on the sampled points.  The Chebyshev basis matrix is precomputed once and reused each iteration.  The coefficient vector $\phi$ is what gets passed to the OCP as a parametric lookup table; the backward pass propagates $\partial J / \partial \phi$ back through the ridge solve to $\partial J / \partial (C_L, C_D, C_M)$.

---

## Optimal Control Problem (`OCP`)

The glider is modeled as a **2D rigid body** (planar flight, pitch as control input) with 8 states: position $(x, z)$, velocity $(v_x, v_z)$, pitch angle $\theta$, pitch rate $\dot\theta$, trim elevator deflection, and adaptive time step $\delta t$.  Aerodynamic forces are evaluated via the Chebyshev reduced model at each collocation node.

The OCP is transcribed as a **direct multiple-shooting NLP** and solved with IPOPT (via CasADi).  Multiple initial conditions are solved in parallel (one process per condition); warm-starting from the previous outer iterate is used to speed up convergence.  Sensitivity of the optimal cost with respect to $\phi$ is extracted analytically via the **IDOC** (Implicit Differentiation of Optimal Control) identities, avoiding finite-difference perturbation of the NLP.

**Evaluation modes:** `Perching` (minimize terminal state deviation from a perch point), `SoftLanding` (similar with a softer target), `Time` (minimize total flight time).

---

## Repository structure

```
glider_optimization/
├── blocks/
│   ├── airfoil.py              # 2D: single-section Kulfan airfoil
│   ├── airfoil3D.py            # 3D: root + tip Kulfan, dynamic centroid
│   ├── neuralFoilSampling.py   # 2D: NeuralFoil (α, Re) grid sampling
│   ├── neuralFoilSampling3D.py # 3D: LLT + NeuralFoil panel solver
│   ├── reducedModel.py         # Chebyshev regression of aero tables
│   ├── ocp.py                  # IPOPT trajectory optimization (CasADi)
│   └── evaluation.py           # Objective + backward through OCP
├── utils/
│   ├── llt.py                  # Differentiable LLT (Picard + implicit diff.)
│   ├── spanwise_geometry.py    # Root/tip interpolation, section centroids
│   ├── glider_jinenv.py        # Glider flight dynamics (CasADi)
│   ├── go_safe_pdp.py          # Constrained OC system (PDP/Safe-PDP)
│   ├── idoc_ineq.py            # IDOC sensitivity identities
│   └── cu_kulfan_airfoil.py    # cuNeuralFoil wrapper
├── config.py                   # Pydantic config schema
├── runner.py                   # Main training loop
└── main.py                     # CLI entry point
conf/
└── test.yaml                   # Example configuration
```

## Quick start

```bash
python -m glider_optimization.main --config conf/test.yaml --run-name my_run
```

Key YAML knobs:

| Field | Effect |
|---|---|
| `neuralFoilSampling.use_3d_llt` | `true` → 3D LLT mode, `false` → 2D NeuralFoil |
| `neuralFoilSampling.n_samples` | Grid size ($N = n^2$ Chebyshev points) |
| `neuralFoilSampling.neuralFoil_size` | NeuralFoil model size (`small` → `xxxlarge`) |
| `reducedModel.chebyshev_degree` | Polynomial degree $d$ |
| `evaluation.mode` | `Perching` / `SoftLanding` / `Time` |
| `plane.wing.*` | Wing planform geometry (span stations, chord, sweep, dihedral) |
