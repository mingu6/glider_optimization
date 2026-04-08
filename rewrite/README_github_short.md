# glider_optimization_lite

Gradient-based nested co-design of airfoil shape and trajectory for agile glider manoeuvres such as perching and soft landing.

This repository couples a shape-dependent aerodynamic model to an inner optimal control problem (OCP) and differentiates the full pipeline end-to-end. For each airfoil design, aerodynamic coefficients are rebuilt, reduced to a local surrogate, embedded in the flight dynamics, and optimized inside a trajectory solver.

## What is in the repo

The code supports two related formulations:

- **2D section-based mode**: NeuralFoil samples are fit with a local Chebyshev surrogate and used inside the OCP.
- **3D finite-wing mode**: a nonlinear lifting-line solver is coupled to a NeuralFoil-based sectional closure before the same reduced-model/OCP pipeline.

The published paper documents the 2D nested co-design methodology. The 3D lifting-line branch is a subsequent code extension built on the same optimization framework.

## 3D aerodynamic model in one paragraph

The 3D branch is a differentiable nonlinear lifting-line method with viscous sectional closure. Root and tip airfoils are optimized independently and interpolated spanwise. Local section coefficients are obtained from a NeuralFoil-based 2D closure evaluated at the local effective angle of attack and Reynolds number. Spanwise circulation is solved by a relaxed fixed-point lifting-line iteration, near-field induced velocity is used to correct local section conditions, and induced drag is treated from a far-wake / Trefftz-plane viewpoint. The resulting aerodynamic tables are then compressed into a local Chebyshev surrogate for the inner OCP.

## Why this structure

The modeling philosophy is to keep the strongest aerodynamic effects that remain affordable inside gradient-based co-design:

- nonlinear viscous section behavior,
- spanwise circulation redistribution,
- induced drag,
- sweep / twist / taper / dihedral effects,
- end-to-end differentiability.

The goal is not to replace CFD, but to provide a physically informed aerodynamic model that is fast enough to sit inside a bilevel optimization loop.

## Current scope and limitations

The current release is **quasi-steady** at sectional level. It does **not** yet include:

- a full dynamic-stall model,
- a fully relaxed free wake,
- or general 3D massively separated-flow physics.

A planned extension is to add low-order unsteady lifting-line force terms together with a lagged effective angle-of-attack state for first-order sectional memory.

## Pipeline

```text
Airfoil / Airfoil3D  →  NeuralFoilSampling / NeuralFoilSampling3D  →  ReducedModel  →  OCP  →  Evaluation
```

## Quick start

```bash
python -m glider_optimization.main --config conf/test.yaml --run-name my_run
```

## Main knobs

- `neuralFoilSampling.use_3d_llt`: switch between 2D and 3D aerodynamic modes
- `neuralFoilSampling.n_samples`: number of aerodynamic samples used for surrogate fitting
- `neuralFoilSampling.neuralFoil_size`: NeuralFoil model size
- `reducedModel.chebyshev_degree`: reduced-model polynomial degree
- `evaluation.mode`: task objective (`Perching`, `SoftLanding`, ...)
- `plane.wing.*`: wing geometry definition

## Reference

If you use the 2D nested co-design methodology, please cite the accompanying paper. The 3D branch extends the same bilevel optimization framework to a lifting-line-coupled finite-wing aerodynamic model.
