"""Augmented Lagrangian handling for the design-level inequality constraints.

Every constraint is written in the form ``g(x) <= 0``.  The augmented Lagrangian
of an inequality constraint (Hestenes / Powell / Rockafellar) is

    psi(x; lam, rho) = 1 / (2 rho) * ( max(0, lam + rho * g(x))^2 - lam^2 )

with the matching multiplier update

    lam <- max(0, lam + rho * g(x)).

Writing ``gh = max(g, -lam / rho)`` for the *shifted* constraint, the same
quantity reads ``psi = lam * gh + rho / 2 * gh^2``, which is what makes it the
inequality counterpart of the familiar equality-constrained formula.  Two
details distinguish it from the naive ``lam * relu(g) + rho / 2 * relu(g)^2``
that is often written instead:

* it is C^1 in x.  The naive form has a gradient that jumps by ``lam`` across
  ``g = 0`` and is identically zero on the whole feasible side, so a multiplier
  that is still active exerts no pull and the iterate parks exactly on the
  boundary.  Here the pull ``lam + rho * g`` decays smoothly and only switches
  off once the constraint is satisfied by the margin ``lam / rho``.
* the ``- lam^2 / (2 rho)`` offset makes ``psi`` the actual value of the
  augmented Lagrangian, so ``J + psi`` is a meaningful merit function.

The multiplier update is likewise the projected one: ``lam`` *decreases* when
the constraint is strictly satisfied and returns to zero for constraints that
end up inactive, instead of ratcheting upwards forever as ``lam += rho *
relu(g)`` does.

The two schedules are deliberately decoupled.  ``lam`` is updated by projected
dual ascent every ``dual_every`` primal steps, which is cheap and is what makes
progress.  ``rho`` is revisited only every ``penalty_every`` dual updates, and
grows only if the infeasibility has not fallen by ``infeasibility_decrease``
*over that whole window*.

Both halves of that matter, and both were learned the hard way here.  Checking
the penalty every step with a "must halve each time" rule is unmeetable by a
single gradient step, so it fires unconditionally: rho went 10 -> its 1e4 cap in
eleven iterations while lam passed 4000, against an objective of order 30.  The
textbook alternative (Conn/Gould/Toint, where a feasibility tolerance ``eta``
decides between updating ``lam`` and raising ``rho``) does not transfer either:
``eta`` tightens by ``/rho**0.9`` per accepted step, which assumes each outer
iteration contains a full inner minimisation.  With one gradient step per outer
iteration ``eta`` collapses to its floor within a few steps and every
noise-level violation then trips the penalty branch -- rho reached 1e4 again on
a two-constraint test whose optimum was perfectly feasible.

A window long enough to average over optimiser noise, with a lenient decrease
target, is what survives contact with a one-step inner loop.

Each constraint carries its own ``lam``, ``rho`` and ``eta`` because the
constraints here live on very different scales (a confidence in [0, 1] against a
Cl/Cd of order ten); this is a diagonal scaling of the penalty and leaves the
method itself unchanged.

Equality constraints ``h(x) = 0`` are supported too, with the classical

    psi = lam * h + rho / 2 * h^2,      lam <- lam + rho * h

(no projection, no shift): declare them by passing the name in ``equalities``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import torch

Number = Union[float, torch.Tensor]


@dataclass
class ConstraintState:
    """Per-constraint dual variable, penalty parameter and infeasibility history."""

    lam: float
    rho: float
    equality: bool = False
    prev_infeasibility: Optional[float] = None


class AugmentedLagrangian:
    """Augmented Lagrangian for a set of named ``g(x) <= 0`` constraints."""

    def __init__(
        self,
        names: Iterable[str],
        equalities: Iterable[str] = (),
        rho: float = 10.0,
        rho_max: float = 1e4,
        rho_factor: float = 2.0,
        infeasibility_decrease: float = 0.75,
        penalty_every: int = 10,
        dual_every: int = 1,
        lam_max: float = 1e6,
    ):
        if rho <= 0.0:
            raise ValueError(f"rho must be positive, got {rho}")
        if rho_factor < 1.0:
            raise ValueError(f"rho_factor must be >= 1, got {rho_factor}")
        if not 0.0 < infeasibility_decrease <= 1.0:
            raise ValueError(f"infeasibility_decrease must be in (0, 1], got {infeasibility_decrease}")
        if penalty_every < 1:
            raise ValueError(f"penalty_every must be >= 1, got {penalty_every}")
        if dual_every < 1:
            raise ValueError(f"dual_every must be >= 1, got {dual_every}")

        self.rho_max = float(rho_max)
        self.rho_factor = float(rho_factor)
        self.infeasibility_decrease = float(infeasibility_decrease)
        self.penalty_every = int(penalty_every)
        self.dual_every = int(dual_every)
        self.lam_max = float(lam_max)

        equalities = set(equalities)
        self.states: Dict[str, ConstraintState] = {
            name: ConstraintState(lam=0.0, rho=float(rho), equality=name in equalities)
            for name in names
        }
        unknown = equalities - set(self.states)
        if unknown:
            raise ValueError(f"equalities not listed in names: {sorted(unknown)}")
        self._primal_steps = 0
        self._dual_updates = 0

    def _state(self, name: str) -> ConstraintState:
        try:
            return self.states[name]
        except KeyError:
            raise KeyError(f"unknown constraint '{name}'; known: {sorted(self.states)}") from None

    def penalty(self, constraints: Mapping[str, Number]) -> torch.Tensor:
        """Sum of ``psi`` over the constraints, differentiable w.r.t. the inputs."""
        total = None
        for name, g in constraints.items():
            st = self._state(name)
            g = g if isinstance(g, torch.Tensor) else torch.as_tensor(float(g))
            if st.equality:
                term = st.lam * g + st.rho / 2.0 * g ** 2
            else:
                term = (torch.relu(st.lam + st.rho * g) ** 2 - st.lam ** 2) / (2.0 * st.rho)
            total = term if total is None else total + term
        return torch.zeros(()) if total is None else total

    def shifted_violation(self, name: str, g: Number) -> float:
        """``max(g, -lam / rho)``: what the AL actually drives to zero."""
        st = self._state(name)
        return float(g) if st.equality else max(float(g), -st.lam / st.rho)

    def step(self, constraints: Mapping[str, Number]) -> None:
        """Account for one primal step, updating duals and penalties on schedule."""
        self._primal_steps += 1
        if self._primal_steps % self.dual_every != 0:
            return
        self._dual_updates += 1
        revisit_penalty = self._dual_updates % self.penalty_every == 0

        for name, g in constraints.items():
            st = self._state(name)
            g_val = float(g)
            infeasibility = abs(self.shifted_violation(name, g_val))

            lam_next = st.lam + st.rho * g_val
            if not st.equality:
                lam_next = max(0.0, lam_next)
            st.lam = max(-self.lam_max, min(self.lam_max, lam_next))

            if revisit_penalty:
                if (
                    st.prev_infeasibility is not None
                    and infeasibility > self.infeasibility_decrease * st.prev_infeasibility
                ):
                    st.rho = min(self.rho_max, self.rho_factor * st.rho)
                st.prev_infeasibility = infeasibility

    def metrics(self, constraints: Optional[Mapping[str, Number]] = None) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, st in self.states.items():
            out[f"lagrangian/lambda_{name}"] = st.lam
            out[f"lagrangian/rho_{name}"] = st.rho
        for name, g in (constraints or {}).items():
            g_val = float(g)
            out[f"lagrangian/violation_{name}"] = abs(g_val) if self._state(name).equality else max(0.0, g_val)
        return out

    def state_dict(self) -> Dict[str, float]:
        return {
            key: value
            for name, st in self.states.items()
            for key, value in (
                (f"lagrangian/lambda_{name}", st.lam),
                (f"lagrangian/rho_{name}", st.rho),
            )
        }

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore from a wandb history row; missing/NaN entries keep their defaults."""

        def read(key: str) -> Optional[float]:
            try:
                value = checkpoint[key]
            except (KeyError, IndexError):
                return None
            if value is None:
                return None
            value = float(value)
            return None if math.isnan(value) else value

        for name, st in self.states.items():
            lam = read(f"lagrangian/lambda_{name}")
            if lam is not None:
                st.lam = lam
            rho = read(f"lagrangian/rho_{name}")
            if rho is not None and rho > 0.0:
                st.rho = rho
