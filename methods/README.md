# Methods

This folder contains the methodological background for the optimization pipeline implemented in this repository.

The goal of these notes is to make the codebase easier to understand without turning the top-level README into a long technical document. The emphasis is on **clarity, traceability to the code, and separation of concerns**:

- the main `README.md` explains **what the repository does**
- the files in `tutorials/` explain **how to run representative examples**
- the files in `methods/` explain **how the underlying models and gradients work**

These notes are written to follow the same design principles as the codebase:

- **KISS**: prefer the simplest model description that still explains the implementation
- **YAGNI**: document what is actually used in the repository, not every possible extension
- **DRY**: define notation and recurring concepts once, then reuse them
- **SOLID**: keep geometry, aerodynamics, reduced-order modeling, and optimal control conceptually separate

---

## Recommended reading order

For most readers, the best order is:

1. [`notation.md`](notation.md)  
   Defines the symbols and conventions used throughout the documentation.

2. [`2d_section_method.md`](2d_section_method.md)  
   Explains the validated 2D pipeline: airfoil parameterization, NeuralFoil sampling, and reduced-order fitting.

3. [`reduced_model.md`](reduced_model.md)  
   Describes how sampled aerodynamic data is compressed into a differentiable surrogate used inside optimization.

4. [`nested_ocp_and_gradients.md`](nested_ocp_and_gradients.md)  
   Explains the nested optimization structure and how derivatives are propagated through the aerodynamic and trajectory subsystems.

5. [`3d_nonlinear_llt_method.md`](3d_nonlinear_llt_method.md)  
   Describes the 3D extension: spanwise geometry mapping, nonlinear lifting-line closure, and implicit differentiation through the fixed-point solve.

If you only care about the currently validated methodology, start with the **2D method** and the **reduced model**.  
If you want to understand the current 3D implementation, read the **2D method first**, then the **3D nonlinear LLT method**.

---

## Documentation map

### `notation.md`
This file defines the common notation used across the repository, including:

- geometric variables
- flight-condition variables
- sectional aerodynamic quantities
- reduced-order model variables
- optimization variables and objectives

This keeps the rest of the documentation compact and avoids redefining the same symbols in multiple places.

---

### `2d_section_method.md`
This file explains the 2D aerodynamic modeling pipeline used as the methodological core of the repository.

Topics covered include:

- airfoil shape parameterization with Kulfan variables
- generation of section geometries from optimization variables
- sampling of aerodynamic coefficients with NeuralFoil
- treatment of the operating domain in angle of attack and Reynolds number
- construction of a smooth surrogate for optimization

This is the best place to start if you want to understand the basic modeling philosophy of the repository.

---

### `reduced_model.md`
This file explains how the sampled aerodynamic data is converted into a reduced-order surrogate suitable for repeated evaluation inside an optimizer.

Topics covered include:

- why a reduced model is needed
- the role of Chebyshev fitting
- surrogate evaluation during optimization
- tradeoffs between fidelity, smoothness, and computational cost

This document sits between the aerodynamic sampling stage and the control/trajectory optimization stage.

---

### `nested_ocp_and_gradients.md`
This file explains the optimization architecture.

Topics covered include:

- outer-loop shape optimization
- inner-loop trajectory optimization
- data flow between geometry, aerodynamics, and dynamics
- gradient propagation across the nested structure
- why implicit or structured differentiation is preferred over brute-force alternatives

This is the key document for understanding the co-design pipeline.

---

### `3d_nonlinear_llt_method.md`
This file explains the current 3D extension of the 2D method.

Topics covered include:

- root-tip airfoil parameterization
- spanwise interpolation of section properties
- finite-wing coupling through a nonlinear lifting-line formulation
- sectional closure using NeuralFoil-derived aerodynamic data
- iterative solution of the lifting-line system
- implicit differentiation through the fixed-point solve

This document is intentionally built on top of the 2D method rather than replacing it. The 3D implementation reuses the same sectional logic, then adds finite-wing coupling and spanwise geometry handling.

---

## Modeling philosophy

The repository is built around the following modeling hierarchy:

1. **Parameterize the geometry**
2. **Sample or evaluate sectional aerodynamics**
3. **Build a differentiable reduced model**
4. **Embed that model inside an optimal control problem**
5. **Differentiate the coupled system with respect to design variables**

The 2D pipeline is the most direct expression of this idea.

The 3D pipeline keeps the same logic, but adds a finite-wing coupling layer that reconstructs local effective flow conditions along the span. In other words, the 3D method is not a separate aerodynamic stack; it is a spanwise-coupled extension of the same 2D constitutive model.

This decomposition is important because it keeps the implementation modular:

- geometry generation is not mixed with trajectory dynamics
- sectional aerodynamics is not mixed with fixed-point solution logic
- reduced-order fitting is not mixed with plotting or reporting
- the optimizer interacts with a compact differentiable interface rather than raw sampled data

---

## Scope of the documentation

These notes document the methods that are actually implemented in the repository.

They do **not** attempt to be a full textbook on:

- airfoil theory
- lifting-line theory in general
- optimal control theory in general
- all possible reduced-order modeling strategies
- all possible unsteady or viscous-inviscid coupling extensions

Where useful, the notes will explain the reasoning behind a modeling choice, but the focus remains on the specific pipeline used here.

---

## Current scope and validation status

At the time of writing, the repository should be interpreted as follows:

- the **2D pipeline** is the methodological core and the main validated path
- the **3D pipeline** is a structured extension that reuses the 2D constitutive logic and adds finite-wing coupling
- the current formulation is primarily **quasi-steady**
- additional unsteady extensions, stronger multi-element interaction models, or higher-fidelity 3D validation are natural future directions, but are not the baseline assumption of the present repository

This distinction is intentional and should remain clear in both the code and the documentation.

---

## Relationship to the code

Each methods page should stay close to the implementation.

The objective is not to reproduce every line of code, but to explain:

- what each subsystem computes
- what its inputs and outputs are
- what mathematical assumptions it encodes
- how it connects to the forward and backward passes of the full pipeline

In that sense, this folder is the conceptual bridge between the code and the tutorials.

---

## Relationship to the tutorials

The `tutorials/` folder answers:

- what to run
- what the inputs are
- what the outputs mean
- how baseline and optimized cases compare

The `methods/` folder answers:

- what equations or approximations are being used
- why the optimization is differentiable
- how 2D and 3D variants are connected
- what assumptions are built into the current formulation

Together, the two folders should make the repository usable both as a **research artifact** and as a **practical optimization tool**.

---

## Suggested next additions

The most useful next files to add are:

- `notation.md`
- `2d_section_method.md`
- `reduced_model.md`
- `nested_ocp_and_gradients.md`
- `3d_nonlinear_llt_method.md`

That order keeps the documentation layered from simplest to most coupled.

---

## Design goal

A good methods section should let a new reader answer three questions quickly:

1. **What is the core model?**
2. **How does the optimizer interact with it?**
3. **Which parts are 2D foundations, and which parts are 3D extensions?**

This folder is intended to answer those questions directly and cleanly.
