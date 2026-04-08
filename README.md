# glider_optimization_lite

Gradient-based nested co-design of airfoil shape and flight trajectory for agile glider manoeuvres such as perching and soft landing.

The repository implements a differentiable bilevel pipeline in which the airfoil geometry is optimized at the outer level, while the inner level solves one or more constrained optimal control problems (OCPs) for the resulting vehicle dynamics. Aerodynamic coefficients are rebuilt from the current shape at every outer iteration, compressed into a local reduced-order model, and then embedded in the trajectory optimizer. Gradients are propagated back through the full chain using analytical, implicit, and automatic differentiation.

## Current scope

This repository contains two related aerodynamic formulations:

- a **2D section-based formulation**, which is the methodology documented in our paper, and
- a **3D lifting-line-coupled extension**, implemented in the codebase as a subsequent development.

The published paper documents the 2D nested co-design formulation and its experimental validation. The 3D branch extends that same co-design logic by coupling a nonlinear sectional aerodynamic closure to a finite-wing lifting-line solver.

The current public code should therefore be read as:

- **validated at paper level in 2D**, and
- **extended in code to 3D with a differentiable nonlinear lifting-line model**.

## Method overview

The optimization pipeline is structured as a differentiable nested program:

\[
\psi \;\longrightarrow\; \text{Aerodynamics} \;\longrightarrow\; \text{Reduced Model} \;\longrightarrow\; \text{OCP} \;\longrightarrow\; \text{Evaluation},
\]

where \(\psi\) denotes the airfoil design parameters.

In the current implementation, the forward pass is organized as the following block chain:

```text
Airfoil / Airfoil3D  →  NeuralFoilSampling / NeuralFoilSampling3D  →  ReducedModel  →  OCP  →  Evaluation
```

At each outer iteration:

1. the current airfoil geometry is built from the optimization variables,
2. aerodynamic coefficients are sampled over a prescribed operating envelope,
3. a local reduced-order surrogate is fit to those samples,
4. the inner trajectory optimization problem is solved using that surrogate inside the dynamics,
5. the resulting task cost is differentiated back to the airfoil parameters.

This follows the same bilevel philosophy as the 2D methodology described in the paper: the aerodynamic model is shape-dependent, rebuilt every iteration, and embedded in a nested trajectory-optimization loop.

## Airfoil parameterization

The airfoil geometry is parameterized using the **Kulfan / CST representation**.

In the 2D formulation, a single airfoil section is optimized.  
In the 3D formulation, independent **root** and **tip** airfoils are optimized and interpolated spanwise.

The design variables are:

- upper-surface Kulfan coefficients,
- lower-surface Kulfan coefficients,
- leading-edge weight,
- trailing-edge thickness.

This yields a smooth, low-dimensional design space suitable for gradient-based optimization while preserving enough flexibility to adapt camber, thickness, and leading-edge shape. The 2D paper uses the same Kulfan-based parameterization at section level.

## 2D mode (`use_3d_llt: false`)

In 2D mode, the aerodynamic model is purely sectional.

For a fixed airfoil design \(\psi\), aerodynamic coefficients are sampled from NeuralFoil over a prescribed \((\alpha, Re)\) envelope:

\[
(\psi,\alpha,Re) \mapsto (C_L, C_D, C_M).
\]

These samples are then fit with a local bivariate Chebyshev surrogate, which is what the OCP queries inside the dynamics. This is the formulation documented in the accompanying paper and should be regarded as the validated methodological core of the repository.

The 2D mode is intentionally simple: it ignores finite-wing effects such as spanwise load redistribution and induced drag, and is best interpreted as a planar section-based co-design model.

## 3D mode (`use_3d_llt: true`)

In 3D mode, the wing is treated as a finite lifting surface with spanwise-varying geometry and nonlinear sectional closure.

The 3D aerodynamic model is a **differentiable nonlinear lifting-line formulation with viscous sectional closure**. The key modeling choice is to keep a high-quality 2D airfoil model at section level and couple it to a lightweight 3D induced-flow solver, rather than attempting to resolve full 3D separated flow inside the optimization loop.

This preserves the dominant finite-wing mechanisms relevant to co-design:

- spanwise circulation redistribution,
- induced velocity and induced angle of attack,
- induced drag,
- sweep, taper, twist, and dihedral effects,
- spanwise variation of the section geometry.

### Root–tip parameterization

The 3D branch optimizes two independent Kulfan airfoils: one at the wing root and one at the wing tip.

At each spanwise station \(\eta = y/(b/2)\), the local section is constructed by interpolation between the root and tip airfoils. This allows the optimizer to exploit spanwise variation in:

- camber,
- thickness,
- leading-edge shape,
- tip unloading or load redistribution.

### Sectional closure with NeuralFoil

At each spanwise station, local aerodynamic coefficients are obtained from a NeuralFoil-based 2D section model evaluated at the local effective angle of attack and Reynolds number.

Conceptually, the 3D solver is not a purely inviscid lifting-line method with a posteriori viscous corrections. Instead, it is a **fixed-point coupling between**:

- a nonlinear viscous sectional model, and
- a 3D induced-flow lifting-line model.

This section-coupled formulation is the core aerodynamic choice of the 3D branch.

### Quarter-chord lifting-line formulation

The 3D lifting-line model is **quarter-chord-centered**.

Circulation is carried on the quarter-chord lifting line, and local flow quantities are evaluated consistently on that same line. This choice is deliberate: it keeps the lifting-line geometry, the sectional closure, and the force/moment bookkeeping internally consistent, and is closer in spirit to modern numerical lifting-line formulations than to Weissinger-style \(1/4\)-vortex, \(3/4\)-collocation schemes.

### LLT fixed-point solve

Given a current spanwise circulation iterate \(\Gamma^{(k)}\), the local effective angle of attack on panel \(i\) is computed as

\[
\alpha_{\mathrm{eff},i}^{(k)}
=
\alpha_{\mathrm{geo},i}
+
\alpha_{\mathrm{twist},i}
-
\alpha_{\mathrm{ind},i}\!\left(\Gamma^{(k)}\right),
\]

where the induced term is obtained from the near-field vortex influence.

The local Reynolds number is then reconstructed from the local speed and local chord,

\[
Re_i^{(k)} = \frac{\rho\,U_i^{(k)}\,c_i}{\mu},
\]

and the sectional closure gives

\[
c_{l,i}^{(k)} = c_l^{2D}\!\left(\alpha_{\mathrm{eff},i}^{(k)}, Re_i^{(k)}, \psi_i\right).
\]

The corresponding unrelaxed circulation update is

\[
\widetilde{\Gamma}_i^{(k+1)}
=
\frac{1}{2}\,U_i^{(k)}\,c_i\,c_{l,i}^{(k)}.
\]

A relaxed Picard iteration is then applied:

\[
\Gamma_i^{(k+1)}
=
(1-\beta)\,\Gamma_i^{(k)}
+
\beta\,\widetilde{\Gamma}_i^{(k+1)}.
\]

Equivalently, in compact form,

\[
\Gamma^{(k+1)}=(1-\beta)\Gamma^{(k)}+\beta\,\mathcal T(\Gamma^{(k)}),
\]

where \(\mathcal T\) is the sectional circulation map induced by the current downwash estimate.

The iteration terminates when the circulation residual falls below tolerance or a maximum iteration count is reached.

### Near-field induced velocity and induced drag

The model treats **local induced-flow correction** and **induced-drag evaluation** as distinct but complementary parts of the 3D aerodynamic formulation.

- **Near-field induced velocity** is used to correct the local section operating point, because this is the quantity needed to update \(\alpha_{\mathrm{eff}}\), Reynolds number, and sectional closure.
- **Induced drag** is evaluated from a far-wake / Trefftz-plane viewpoint, where lift-induced drag is naturally defined at wing level rather than as a purely local section quantity.

This separation avoids conflating viscous profile drag with global lift-induced drag and makes the 3D formulation more physically consistent.

### Sweep, twist, dihedral, and dynamic geometry

The 3D branch supports:

- **sweep**, through projection of local section quantities onto the lifting-surface-aligned frame,
- **twist**, through spanwise geometric incidence variation,
- **dihedral**, through out-of-plane displacement of the lifting-line geometry and control points,
- **dynamic centroid / reference-geometry updates**, when enabled, so that moment arms and aerodynamic reference quantities remain consistent with the current optimized airfoil geometry.

## Reduced aerodynamic model (`ReducedModel`)

For a fixed airfoil design, the sampled aerodynamic coefficients are approximated over the operating envelope by a tensor-product Chebyshev polynomial:

\[
C(\alpha, Re)
\approx
\sum_{i=0}^{d}\sum_{j=0}^{d}
\phi_{ij}\,T_i(\bar\alpha)\,T_j(\overline{Re}).
\]

This reduced model is fit separately for \(C_L\), \(C_D\), and \(C_M\).

The purpose of this step is purely computational: it replaces repeated evaluation of the underlying aerodynamic model with a local differentiable polynomial surrogate that is cheap enough to embed in the inner OCP.

This is the same reduced-order logic used in the 2D paper: the neural or lifting-line-based aerodynamic model remains the constitutive law, while the Chebyshev approximation is only a local optimization-time surrogate.

## Optimal Control Problem (`OCP`)

The inner problem is a constrained trajectory optimization problem for planar glider flight.

At the physical-model level, the glider is modeled as a planar rigid body with position, pitch attitude, elevator angle, and associated velocities, while the control is the elevator rate. In the implementation, the transcription may also include auxiliary decision variables related to the time discretization.

Aerodynamic forces and moments are evaluated at each trajectory node using the local Chebyshev surrogate built from the current design. The OCP is transcribed as a nonlinear program and solved with IPOPT through CasADi.

Sensitivities are propagated analytically / implicitly through the optimal-control layer rather than by finite differences, following the same nested-co-design rationale as in the 2D methodology.

## Differentiation strategy

The repository is designed so that gradients can flow from the task objective all the way back to the airfoil parameters.

The main differentiation mechanisms are:

- **automatic differentiation** through explicitly differentiable blocks,
- **analytic gradients** through the reduced-model fit,
- **implicit differentiation** through converged fixed-point solves and optimal-control conditions, rather than unrolling every inner iteration.

In the 3D branch, the converged lifting-line fixed point is differentiated as a fixed-point solution rather than as a long unrolled iterative graph.

## Planned unsteady extension

The current release is **quasi-steady** at sectional level: at each trajectory node, sectional loads are evaluated from an instantaneous effective angle of attack and Reynolds number through a static sectional map.

A planned extension is to enrich the model in two complementary low-order ways.

### 1. Unsteady lifting-line force terms

At lifting-line level, the intended formulation follows Sugar-Gabor-style unsteady nonlinear lifting-line modeling by augmenting the steady force law with contributions associated with:

- time variation of circulation, and
- kinematic variation of the local section frame.

These terms are intended to capture non-circulatory and rapid-circulation effects associated with aggressive manoeuvres.

### 2. Lagged effective angle of attack

At sectional level, the intended first approximation is a lagged effective-angle-of-attack state \(\alpha_f\), governed by

\[
\dot{\alpha}_f = \frac{\alpha-\alpha_f}{\tau_\alpha}.
\]

The static sectional surrogate is then queried at \(\alpha_f\) rather than at the instantaneous \(\alpha\).

This is a reduced first-order memory model inspired by low-order dynamic-stall formulations, in which the effective angle of attack lags the geometric angle of attack. A convective scaling of the form

\[
\tau_\alpha = k_\alpha \frac{c}{2V}
\]

is the intended starting point, where \(c\) is a reference chord, \(V\) is the local section speed, and \(k_\alpha\) is a dimensionless tuning constant.

### Scope of the unsteady model

These unsteady components are **planned future work** and are **not yet part of the current release**.

The intended extension should be interpreted as a lightweight first-order unsteady enrichment compatible with nested gradient-based co-design. It is not claimed to be a full dynamic-stall model or a fully relaxed unsteady-wake solver.

## Confidence handling and surrogate validity

NeuralFoil exposes an analysis-confidence signal. In this framework, that quantity is interpreted as a **trust metric for the static sectional surrogate**, not as a direct physical separation variable.

The optimization therefore constrains the design to remain inside regions where the surrogate is sufficiently reliable. This is an important part of the methodology: without such constraints, gradient-based optimization can exploit low-trust regions of the aerodynamic model and converge to aphysical shapes. The 2D paper documents this point explicitly.

## What this model is, and what it is not

Taken together, the 3D methodology should be read as:

- a **differentiable nonlinear lifting-line model**,
- with **viscous 2D sectional closure**,
- **near-field induced-flow correction**,
- **far-field induced-drag evaluation**,
- and a **local reduced-order surrogate** embedded in a nested co-design loop.

It is intended to provide the best aerodynamic fidelity that can reasonably be embedded inside a gradient-based bilevel optimization pipeline.

It is **not** currently:

- a full CFD solver,
- a full free-wake / relaxed-wake solver,
- a full dynamic-stall model with dedicated vortex or separation states,
- or a general 3D massively separated-flow model.

These limitations are deliberate and should be read as part of the modeling scope, not as hidden assumptions.

## Repository structure

```text
glider_optimization/
├── blocks/
│   ├── airfoil3D.py
│   ├── neuralFoilSampling.py
│   ├── neuralFoilSampling3D.py
│   ├── reducedModel.py
│   ├── ocp.py
│   └── evaluation.py
├── utils/
│   ├── llt.py
│   ├── spanwise_geometry.py
│   ├── glider_jinenv.py
│   ├── go_safe_pdp.py
│   ├── idoc_ineq.py
│   └── cu_kulfan_airfoil.py
├── config.py
├── runner.py
└── main.py

conf/
└── test.yaml
```

## Quick start

```bash
python -m glider_optimization.main --config conf/test.yaml --run-name my_run
```

## Key configuration knobs

- `neuralFoilSampling.use_3d_llt`  
  `true` enables the 3D lifting-line-coupled aerodynamic model; `false` uses the 2D sectional formulation.

- `neuralFoilSampling.n_samples`  
  Number of aerodynamic sample points used to build the local reduced-order surrogate.

- `neuralFoilSampling.neuralFoil_size`  
  NeuralFoil model size used for sectional predictions.

- `reducedModel.chebyshev_degree`  
  Degree of the Chebyshev polynomial surrogate.

- `evaluation.mode`  
  Task objective, e.g. `Perching` or `SoftLanding`.

- `plane.wing.*`  
  Wing planform definition, including span stations, chord distribution, sweep, twist, and dihedral.

## Reference

If you use the 2D nested co-design methodology, please cite the accompanying paper. The repository README and code extend that same framework to a 3D lifting-line-coupled aerodynamic formulation.
