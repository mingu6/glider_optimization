# Aero-ROM: Differentiable 3D Aerodynamics with LLT + cuNeuralFoil

This repository implements a **differentiable 3D aerodynamic reduced-order model (ROM)** based on a **Weissinger–L lifting-line theory (LLT)** coupled with **cuNeuralFoil** for sectional aerodynamics.

The code is designed for **gradient-based shape optimization and optimal control**, with a strong emphasis on:
- correctness,
- clarity,
- and minimal abstraction (KISS / YAGNI / SIMPLE).

Two differentiation strategies are provided:
1. **Explicit (unrolled) differentiation**
2. **Implicit differentiation (IFT / adjoint)**

Both operate directly on **Kulfan airfoil shape parameters**.

---

## Features

- 3D aerodynamic model using Weissinger–L LLT
- Section aerodynamics from **cuNeuralFoil**
- Full differentiation w.r.t. **Kulfan parameters**
- Explicit and implicit (adjoint) gradient computation
- One-shot pipeline producing:
  - CSV aerodynamic coefficient surfaces
  - Reusable PyTorch differentiable blocks
- Backend support: CPU / MPS / CUDA  
  (implicit adjoint solve uses CPU fallback on MPS)

---

## Repository Structure

```text
aero_rom/
├── src/
│   ├── diff_llt.py           # Explicit (unrolled) differentiable LLT
│   ├── implicit_llt.py       # Implicit LLT via IFT / adjoint
│   ├── implicit_models.py    # CL / CD / CM coefficient blocks
│   ├── diff_pipeline.py      # One-shot pipeline (dataset + blocks)
│   ├── load_blocks.py        # Reload saved blocks
│   └── pipeline.py           # Non-differentiable baseline pipeline
│
├── run_from_config.py        # CLI entry point
├── data/
│   └── config.json           # Geometry + flow configuration
│
├── artifacts/
│   ├── raw_surfaces/         # CSV coefficient grids
│   └── models/               # Serialized PyTorch blocks
│
└── README.md
```
---

## Mathematical Model

### Forward problem (LLT)

The circulation distribution Γ is obtained as the fixed point of a nonlinear operator:

Γ = G(Γ, θ)

where:
- θ are the Kulfan airfoil parameters,
- G combines induced velocities (LLT) and sectional aerodynamics (cuNeuralFoil).

Once converged, global coefficients (CL, CD, CM) are computed by spanwise integration.

---

## Differentiation Strategies

### 1. Explicit (Unrolled) Differentiation

- LLT iterations are fully unrolled.
- Gradients are computed directly via PyTorch autograd.
- Simple and robust.

**Pros**
- Works on all backends (CPU / MPS / CUDA)
- Easy to debug and validate
- Ideal reference for finite-difference checks

**Cons**
- Backward cost scales with number of LLT iterations
- Large autograd graphs

Implemented in: `src/diff_llt.py`

---

### 2. Implicit Differentiation (IFT / Adjoint)

The LLT solution is treated as the root of a residual:

F(Γ, θ) = Γ − G(Γ, θ) = 0

Using the Implicit Function Theorem:

dL/dθ = ∂L/∂θ − λᵀ ∂F/∂θ  
with  
(∂F/∂Γ)ᵀ λ = ∂L/∂Γ

**Key design choices**
- Forward LLT solve runs under `torch.no_grad()`
- Backward reconstructs a differentiable residual
- Exact Jacobian ∂F/∂Γ via autograd
- Exact linear solve for adjoint system
- CPU fallback for linear solve on MPS

Implemented in: `src/implicit_llt.py`
---

## Typical Workflow

### Run the pipeline once

```bash
python run_from_config.py data/config.json
```

### Reload and use differentiable blocks

```python
from src.load_blocks import load_blocks_from_ckpt
blocks = load_blocks_from_ckpt("artifacts/models/3d_blocks.pt", device="cuda")
cl_block = blocks["wing"]["cl_block"]
CL = cl_block(5.0, 18.0)
grads = cl_block.backward(5.0, 18.0)
```

---

## Backend Notes

- CUDA: fully supported
- MPS: adjoint linear solve uses CPU fallback
- CPU: supported but slower


