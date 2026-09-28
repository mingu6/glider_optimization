"""Correctness checks for the augmented Lagrangian.

Each case is a problem whose optimum *and* KKT multiplier are known in closed
form, solved the way the pipeline solves: gradient steps on the augmented
Lagrangian with the duals updated by ``step()``.

Step sizes matter here. The AL penalty contributes curvature ``rho * ||grad g||^2``,
so a constraint whose Jacobian is large needs a proportionally smaller step --
divergence in these tests usually means the step size, not the multiplier
schedule. Adam is a poor choice for these 1-D problems for a related reason: it
normalises away the gradient's magnitude and so keeps taking full-size steps
across the ``relu`` kink at the constraint boundary instead of settling onto it.
"""

import torch

from glider_optimization.utils.augmented_lagrangian import AugmentedLagrangian


def _descend(al, objective, constraints, n=1, lr=0.05, iters=4000):
    x = torch.zeros(n, requires_grad=True)
    for _ in range(iters):
        loss = objective(x) + al.penalty(constraints(x))
        (grad,) = torch.autograd.grad(loss, [x])
        with torch.no_grad():
            x -= lr * grad
        al.step({k: float(v.detach()) for k, v in constraints(x).items()})
    return x.detach()


def test_active_inequality_recovers_optimum_and_multiplier():
    # min -x  s.t.  x <= 1   ->   x* = 1, lambda* = 1
    al = AugmentedLagrangian(names=("c",), rho=10.0, dual_every=1)
    x = _descend(al, lambda x: -x[0], lambda x: {"c": x[0] - 1.0})
    assert abs(float(x[0]) - 1.0) < 1e-3
    assert abs(al.states["c"].lam - 1.0) < 5e-2
    assert al.states["c"].rho <= 100.0  # no runaway escalation on a feasible problem


def test_inactive_inequality_leaves_multiplier_and_penalty_alone():
    # min (x - 1/2)^2  s.t.  x <= 1   ->   x* = 1/2, constraint slack, lambda* = 0
    al = AugmentedLagrangian(names=("c",), rho=10.0, dual_every=1)
    x = _descend(al, lambda x: (x[0] - 0.5) ** 2, lambda x: {"c": x[0] - 1.0})
    assert abs(float(x[0]) - 0.5) < 1e-3
    assert al.states["c"].lam == 0.0
    assert al.states["c"].rho == 10.0


def test_equality_recovers_optimum_and_signed_multiplier():
    # min x^2  s.t.  x = 2   ->   x* = 2, lambda* = -4
    al = AugmentedLagrangian(names=("h",), equalities=("h",), rho=10.0, dual_every=1)
    x = _descend(al, lambda x: x[0] ** 2, lambda x: {"h": x[0] - 2.0}, lr=0.02, iters=6000)
    assert abs(float(x[0]) - 2.0) < 1e-3
    assert abs(al.states["h"].lam + 4.0) < 5e-2


def test_multiplier_is_released_when_constraint_goes_slack():
    """The projected update must let go -- `lam += rho * relu(g)` never does."""
    al = AugmentedLagrangian(names=("c",), rho=10.0, dual_every=1)
    al.states["c"].lam = 7.0
    for _ in range(50):
        al.step({"c": -0.5})
    assert al.states["c"].lam == 0.0
    assert al.states["c"].rho == 10.0


def test_infeasible_constraint_escalates_penalty_with_bounded_multiplier():
    """Driving rho to its cap is how an AL reports infeasibility."""
    al = AugmentedLagrangian(names=("c",), rho=10.0, rho_max=1e4, dual_every=1)
    for _ in range(400):
        al.step({"c": 0.5})
    assert al.states["c"].rho == 1e4
    assert al.states["c"].lam <= al.lam_max


def test_constraints_on_different_scales_converge_independently():
    # min -x0 - x1  s.t.  x0 <= 1/100, x1 <= 100   ->  both active, lambda* = (1, 1)
    al = AugmentedLagrangian(names=("a", "b"), rho=10.0, dual_every=1)
    constraints = lambda x: {"a": x[0] - 0.01, "b": x[1] - 100.0}
    x = _descend(al, lambda x: -x[0] - x[1], constraints, n=2, lr=0.01, iters=60000)
    assert abs(float(x[0]) - 0.01) < 1e-4
    assert abs(float(x[1]) - 100.0) < 1e-2
    assert abs(al.states["a"].lam - 1.0) < 2e-2
    assert abs(al.states["b"].lam - 1.0) < 2e-2
