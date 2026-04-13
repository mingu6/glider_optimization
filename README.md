# Glider Optimization

Gradient-based nested co-design of aerodynamic shape and flight trajectory for agile glider manoeuvres such as perching and soft landing.

This repository implements a differentiable bilevel pipeline in which the **outer loop** updates the airfoil or wing shape, while the **inner loop** solves a constrained optimal control problem (OCP) for the resulting vehicle dynamics. At each outer iteration, aerodynamic coefficients are rebuilt from the current geometry, compressed into a local reduced-order model, and embedded inside the trajectory optimizer. Gradients are then propagated back through the full chain using automatic, analytic, and implicit differentiation.

## Why this repository exists

The goal of this project is to make **shape-and-trajectory co-design** practical for agile gliders without resorting to a full CFD-in-the-loop workflow.

Instead of optimizing trajectory with a fixed aerodynamic model, or optimizing shape for a fixed flight condition, this repository couples both:

- **shape changes** modify the aerodynamic model,
- the **trajectory optimizer** reacts to those aerodynamic changes,
- and the resulting **task-level objective** is differentiated all the way back to the geometry parameters.

The result is a lightweight but expressive pipeline for studying how aerodynamic design and flight strategy should be optimized together.

## Current scope

The repository currently contains two related aerodynamic formulations:

- a **2D section-based formulation**, which is the methodology documented and validated in our paper, and
- a **3D lifting-line-coupled extension**, implemented in the codebase as a subsequent development.

The code should therefore be read as:

- **validated at paper level in 2D**, and
- **extended in code to 3D with a differentiable nonlinear lifting-line model**.

That distinction is important. The 2D formulation is the published methodological core. The 3D branch follows the same nested co-design logic, but extends the aerodynamic model to a finite wing with spanwise-varying geometry and induced-flow coupling.

## High-level pipeline

The optimization chain is organized as:

```text
Airfoil / Airfoil3D
    → NeuralFoilSampling / NeuralFoilSampling3D
    → ReducedModel
    → OCP
    → Evaluation
```

At each outer iteration:

1. the current airfoil or wing geometry is built from the design variables,
2. aerodynamic coefficients are sampled over a prescribed operating envelope,
3. a local reduced-order surrogate is fit to those samples,
4. the inner trajectory optimization problem is solved using that surrogate,
5. the resulting task cost is differentiated back to the geometry.

This keeps the optimization loop fully shape-aware while remaining computationally lightweight enough for repeated outer iterations.

## Aerodynamic modeling philosophy

### 2D mode

In 2D mode, the aerodynamic model is purely sectional. The airfoil is parameterized with **Kulfan / CST coefficients**, sampled with **NeuralFoil** over an `(alpha, Re)` envelope, and then approximated locally with a **Chebyshev reduced-order model** used inside the OCP.

This is the simplest and most validated formulation in the repository.

### 3D mode

In 3D mode, the wing is represented by **root and tip Kulfan airfoils** interpolated spanwise and coupled to a **nonlinear lifting-line solver** with viscous sectional closure.

The 3D branch is designed to capture the main finite-wing effects relevant to co-design:

- spanwise circulation redistribution,
- induced velocity and induced angle of attack,
- induced drag,
- sweep, taper, twist, and dihedral effects,
- spanwise variation of section geometry.

Rather than embedding a full 3D flow solver in the loop, the repository combines a high-quality 2D sectional model with a lightweight 3D induced-flow model. That tradeoff is deliberate: it keeps the method differentiable, fast, and useful for optimization.

## What this model is, and what it is not

This repository is intended to provide the highest aerodynamic fidelity that can reasonably be embedded inside a gradient-based nested co-design pipeline.

It is:

- a **differentiable aerodynamic co-design framework**,
- with **NeuralFoil-based sectional closure**,
- a **local reduced-order surrogate** for optimization-time efficiency,
- and a **3D nonlinear lifting-line extension** for finite-wing studies.

It is not:

- a full CFD solver,
- a full free-wake or relaxed-wake solver,
- a full dynamic-stall model,
- or a general-purpose massively separated 3D flow solver.

These limitations are deliberate and should be read as part of the modeling scope, not as hidden assumptions.

## Repository structure

The repository is organized around three complementary layers:

```text
glider_optimization/
├── glider_optimization/   # core package
│   ├── blocks/            # differentiable pipeline blocks
│   ├── utils/             # lifting-line, geometry, and utility functions
│   ├── config.py
│   ├── runner.py
│   └── main.py
├── conf/                  # run configurations
├── tutorials/             # reproducible example studies
└── methods/               # mathematical background and modeling notes
```

The **core package** contains the implementation.
The **tutorials** folder is intended to expose reproducible example runs.
The **methods** folder is intended to carry the detailed modeling and mathematical background that would otherwise make this front page too dense.

## Installation

Clone the repository and install it in editable mode:

```bash
pip install -e .
```

Typical dependencies include:

- `torch`
- `casadi`
- `aerosandbox`
- `neuralfoil`
- `numpy`
- `matplotlib`
- `pandas`
- `PyYAML`

## Quick start

Run a configuration with either the module entry point:

```bash
python -m glider_optimization.main --config conf/test.yaml --run-name demo_run
```

or the installed console script:

```bash
glider-opt --config conf/test.yaml --run-name demo_run
```

## Key configuration knobs

A few settings determine most of the behavior:

- `neuralFoilSampling.use_3d_llt`
  - `false`: use the 2D sectional formulation
  - `true`: use the 3D lifting-line-coupled formulation

- `neuralFoilSampling.n_samples`
  - number of aerodynamic samples used to build the local reduced-order model

- `neuralFoilSampling.neuralFoil_size`
  - NeuralFoil model size used for sectional predictions

- `reducedModel.chebyshev_degree`
  - degree of the Chebyshev surrogate used inside the OCP

- `evaluation.mode`
  - task objective, for example `Perching` or `SoftLanding`

- `plane.wing.*`
  - finite-wing geometry definition, including span stations, chord, sweep, twist, and dihedral

## Planned documentation structure

To keep the repository clean and discoverable, the project is best read through two entry points:

- `tutorials/` for runnable examples and baseline-versus-optimized comparisons,
- `methods/` for the mathematical background, assumptions, and derivations.

The intended tutorial set is:

1. **2D shape optimization under fixed conditions**
2. **3D wing shape optimization under fixed conditions**
3. **2D trajectory + shape co-design**
4. **3D trajectory + shape co-design**

The intended methods documentation is split by topic rather than concentrated in one long README:

- 2D sectional method
- 3D nonlinear lifting-line method
- reduced-order aerodynamic model
- nested OCP and gradient propagation
- assumptions, notation, and limitations

## Validation status and limitations

The repository is strongest when interpreted as follows:

- the **2D nested co-design methodology** is the validated foundation,
- the **3D aerodynamic branch** is a differentiable finite-wing extension built in the same spirit,
- the current release is **quasi-steady** at sectional level,
- and unsteady extensions remain future work.

That means the code is already useful for research and design-space exploration, but the aerodynamic scope should be interpreted honestly.

## Citation

If you use this repository in academic work, please cite the accompanying paper for the 2D nested co-design methodology and describe the 3D lifting-line-coupled branch as a subsequent code-level extension.

A BibTeX entry can be added here once the final reference is fixed.

## License

Add your preferred license here before public release.
