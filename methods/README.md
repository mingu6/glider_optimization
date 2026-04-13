# Methods

This folder contains the background documentation for the models used in this repository.

## Contents

- `notation.md`  
  Common notation used across the documentation.

- `2d_section_method.md`  
  2D airfoil parameterization, NeuralFoil sampling, and surrogate construction.

- `reduced_model.md`  
  Reduced-order representation used inside the optimization pipeline.

- `nested_ocp_and_gradients.md`  
  Nested trajectory and shape optimization, including gradient flow through the coupled system.

- `3d_nonlinear_llt_method.md`  
  3D wing formulation based on spanwise geometry mapping and nonlinear lifting-line coupling.

## Scope

The methodological core of the repository is the 2D pipeline. The 3D formulation extends the same sectional modeling logic to finite wings through spanwise coupling.

The current implementation is primarily quasi-steady.

## Relation to the repository

- `README.md` gives an overview of the repository
- `tutorials/` contains runnable examples
- `methods/` explains the underlying models
