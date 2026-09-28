from ..blockBase import Block
from typing import override
from ..config import Config, EvaluationMode
from ..utils.augmented_lagrangian import AugmentedLagrangian
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import wandb
import logging

EVALUATION = "Trajectory"

class Evaluation(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging
        self.objective_evolution = []
        self.cost_evolution = []
        self.last_traj = None
        self.eval_map = {
            EvaluationMode.Perching: {
                "fwd": self.forward_ocp_cost,
                "bwd": self.backward_ocp_cost
            },
            EvaluationMode.SoftLanding: {
                "fwd": self.forward_ocp_cost,
                "bwd": self.backward_ocp_cost
            },
            EvaluationMode.Time: {
                "fwd": self.forward_time,
                "bwd": self.backward_time
            },
            EvaluationMode.Distance: {
                "fwd": self.forward_distance,
                "bwd": self.backward_distance
            },
            EvaluationMode.ShortLanding: {
                "fwd": self.forward_landing_distance,
                "bwd": self.backward_landing_distance
            },
            EvaluationMode.SoftTouchdown: {
                "fwd": self.forward_touchdown_energy,
                "bwd": self.backward_touchdown_energy
            }
        }
        self._extra_metrics = {}

        # epsilon-constraint co-design: minimise Phi_1 subject to Phi_j <= eps_j.
        # Each bound is in its own task's units (metres here), so nothing needs
        # normalising against anything else, and sweeping eps traces the Pareto front
        # including non-convex stretches that a weighted sum of task costs cannot
        # reach. The requirements ride on the same augmented Lagrangian as the design
        # constraints, since `Phi_j - eps_j <= 0` is just another g(psi) <= 0.
        self.tasks = config.task_list
        self.requirements = self.tasks[1:]
        self.task_costs, self.task_constraints, self._last_by_task = {}, {}, {}
        nf = config.neuralFoilSampling
        self.task_al = AugmentedLagrangian(
            names=[t.name for t in self.requirements],
            equalities=[t.name for t in self.requirements
                        if getattr(t, "epsilon_active", False)],
            rho=nf.task_rho if nf.task_rho is not None else nf.rho,
            rho_max=nf.task_rho_max if nf.task_rho_max is not None else nf.rho_max,
            rho_factor=nf.rho_factor,
            infeasibility_decrease=nf.infeasibility_decrease,
            penalty_every=nf.penalty_every, dual_every=nf.dual_every,
        )
        
    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self.last_traj = downstream_info["trajectory"]
        self._extra_metrics = {}
        self._last_by_task = downstream_info["trajectory_by_task"]

        # Phi_j for every task, each evaluated on its own trajectories and in its own units
        self.task_costs = {}
        for task in self.tasks:
            self.last_traj = self._last_by_task[task.name]
            self._current_task = task.name
            self.task_costs[task.name] = self.eval_map[task.mode]["fwd"]()

        J = self.task_costs[self.tasks[0].name]
        self.task_constraints = {
            t.name: self.task_costs[t.name] - t.epsilon for t in self.requirements
        }
        task_aug = float(self.task_al.penalty(self.task_constraints)) if self.requirements else 0.0

        aug = downstream_info["augmented_lagrangian"]
        total_obj = J + float(aug) + task_aug

        for t in self.requirements:
            self._extra_metrics[f"tasks/{t.name}_cost"] = self.task_costs[t.name]
            self._extra_metrics[f"tasks/{t.name}_margin"] = t.epsilon - self.task_costs[t.name]
        self._extra_metrics["tasks/primary_cost"] = J
        self._extra_metrics["evaluation/task_augmented_lagrangian"] = task_aug
        if self.requirements:
            self._extra_metrics.update(self.task_al.metrics(self.task_constraints))

        iteration = downstream_info["iteration"]
        if iteration % self.config.io.log_every == 0:
            summary = ", ".join(
                f"{t.name}={self.task_costs[t.name]:.3f}"
                + (f" (eps {t.epsilon:.3f})" if t.epsilon is not None else "")
                for t in self.tasks
            )
            self.logger.info(f"Objective (total) = {total_obj}, Cost = {J} | {summary}")
        
        if self.config.io.wandb.enabled:
            self._log_to_wandb(total_obj, J, aug, iteration)
        else:
            self.objective_evolution.append(total_obj)
            self.cost_evolution.append(J)

        return {
            "total_obj": total_obj
        }
    
    def backward(self, upstream_grads: Dict[str, Any]) -> Dict[str, Any]:
        dJ_deps_by_task = {}
        for task in self.tasks:
            self.last_traj = self._last_by_task[task.name]
            self._current_task = task.name
            dJ_deps_by_task[task.name] = self.eval_map[task.mode]["bwd"]()

        # d(total)/d(Phi_j): 1 for the primary objective; for a requirement it is the
        # derivative of its AL term.
        #
        # Inequality: relu(lam_j + rho_j (Phi_j - eps_j)) -- zero while the requirement
        # is comfortably met, so a slack task costs no gradient at all.
        #
        # Equality: lam_j + rho_j (Phi_j - eps_j), WITHOUT the relu. Clamping it would
        # zero the gradient exactly when Phi_j drifts below eps_j, leaving nothing to
        # pull the design back onto the bound -- which is what happened: with the relu
        # in place an "active" bound still let Phi_2 fall from 201.7 to 195.4, and
        # raising rho 20x changed nothing because 20 * 0 is still 0.
        task_weights = {self.tasks[0].name: 1.0}
        for t in self.requirements:
            st = self.task_al.states[t.name]
            raw = st.lam + st.rho * self.task_constraints[t.name]
            task_weights[t.name] = raw if st.equality else max(0.0, raw)
            self.logger.info(
                f"[AL] {t.name}: g={self.task_constraints[t.name]:+.3f} lam={st.lam:+.3f} "
                f"rho={st.rho:.1f} raw={raw:+.3f} weight={task_weights[t.name]:+.3f} "
                f"equality={st.equality}")

        if self.requirements:
            self.task_al.step(self.task_constraints)

        return {"dJ_deps_by_task": dJ_deps_by_task, "task_weights": task_weights}

    def forward_ocp_cost(self):
        cost_vals = [float(t["cost"][0][0]) for t in self.last_traj if t["success"]]
        if not cost_vals:
            self.logger.critical("All trajectories failed; objective is undefined this iteration")
            return float("nan")
        return sum(cost_vals) / len(cost_vals)

    def forward_distance(self):
        """Horizontal distance covered, negated: the pipeline minimises its objective.

        The design gradient reaches this through the terminal x, which the inner OCP's
        sensitivity does see -- unlike a term on the timestep state, whose derivative
        w.r.t. the design is structurally zero.
        """
        successful = [t for t in self.last_traj if t["success"]]
        if not successful:
            self.logger.critical("All trajectories failed; objective is undefined this iteration")
            return float("nan")

        distances = [float(t["state_traj_opt"][-1, 0] - t["state_traj_opt"][0, 0]) for t in successful]
        altitudes = [float(t["state_traj_opt"][-1, 1]) for t in successful]
        flight_times = [float(t["state_traj_opt"][-1, 7]) * (t["state_traj_opt"].shape[0] - 1) for t in successful]

        mean_distance = sum(distances) / len(distances)
        mean_altitude = sum(altitudes) / len(altitudes)
        mean_time = sum(flight_times) / len(flight_times)
        tag = getattr(self, "_current_task", "distance")
        self._extra_metrics.update({
            f"evaluation/{tag}_glide_distance": mean_distance,
            # touchdown altitude: the terminal penalty is soft, so this is the check
            # that the distance was actually flown out to the ground
            f"evaluation/{tag}_touchdown_altitude": mean_altitude,
            f"evaluation/{tag}_flight_time": mean_time,
        })
        self.logger.info(
            f"[{tag}] glide distance = {mean_distance:.3f} m, "
            f"touchdown altitude = {mean_altitude:.4f} m, flight time = {mean_time:.3f} s"
        )
        return -mean_distance

    def forward_touchdown_energy(self):
        """Kinetic energy carried into the ground, in joules, minimised.

        Unlike landing distance this is a terminal *state* quantity, so it moves
        smoothly with the design rather than jumping when the free final time drops
        the touchdown into a different local optimum.
        """
        successful = [t for t in self.last_traj if t["success"]]
        if not successful:
            self.logger.critical("All trajectories failed; objective is undefined this iteration")
            return float("nan")

        m = self.config.ocp.glider_mass
        energies, speeds, altitudes = [], [], []
        for t in successful:
            xT = t["state_traj_opt"][-1]
            v_sq = float(xT[4]) ** 2 + float(xT[5]) ** 2
            energies.append(0.5 * m * v_sq)
            speeds.append(v_sq ** 0.5)
            altitudes.append(float(xT[1]))

        mean_energy = sum(energies) / len(energies)
        mean_speed = sum(speeds) / len(speeds)
        mean_altitude = sum(altitudes) / len(altitudes)
        tag = getattr(self, "_current_task", "softtouchdown")
        self._extra_metrics.update({
            f"evaluation/{tag}_touchdown_energy": mean_energy,
            f"evaluation/{tag}_touchdown_speed": mean_speed,
            f"evaluation/{tag}_touchdown_altitude": mean_altitude,
        })
        self.logger.info(
            f"[{tag}] touchdown energy = {mean_energy:.4f} J, "
            f"speed = {mean_speed:.3f} m/s, altitude = {mean_altitude:.4f} m"
        )
        return mean_energy

    def backward_touchdown_energy(self):
        """d/dx_T of (1/2) m (xdot^2 + zdot^2), averaged over trajectories."""
        m = self.config.ocp.glider_mass
        successful = [t for t in self.last_traj if t["success"]]
        n = max(len(successful), 1)
        grads = []
        for traj in self.last_traj:
            g = np.zeros(traj["state_traj_opt"].shape)
            if traj["success"]:
                xT = traj["state_traj_opt"][-1]
                g[-1, 4] = m * float(xT[4])
                g[-1, 5] = m * float(xT[5])
            grads.append(g)
        return grads

    def forward_landing_distance(self):
        """Distance to touchdown, minimised -- the same quantity as forward_distance
        but reported with its natural sign, so an epsilon bound reads in metres."""
        return -self.forward_distance()

    def backward_landing_distance(self):
        return [-d for d in self.backward_distance()]

    def backward_distance(self):
        # Must mirror forward_distance, which is a *mean over successful* trajectories.
        # Handing back -1.0 per trajectory instead of -1/n made the gradient n times
        # too large, and -- worse -- gave failed solves a full-magnitude gradient built
        # from their garbage state_traj_opt. With 1-2 of 4 solves failing routinely,
        # a large share of the design direction was coming from failed trajectories.
        successful = [t for t in self.last_traj if t["success"]]
        n = max(len(successful), 1)

        dJ_deps_list = []
        for traj in self.last_traj:
            dJ_deps_traj = np.zeros(traj['state_traj_opt'].shape)
            if traj["success"]:
                # ocp.py divides by the successful count; do not divide again here
                dJ_deps_traj[-1, 0] = -1.0        # d(-x_T)/dx_T
            dJ_deps_list.append(dJ_deps_traj)

        return dJ_deps_list

    def forward_time(self):
        total_time = [t["state_traj_opt"][:,7].sum() for t in self.last_traj if t["success"]]
        if not total_time:
            self.logger.critical("All trajectories failed; objective is undefined this iteration")
            return float("nan")
        return sum(total_time) / len(total_time)
    
    def backward_ocp_cost(self):
        w = self.config.ocp.terminal_state_weight

        # As in backward_distance: average over successful trajectories only, so a
        # failed solve contributes nothing rather than a gradient read off garbage.
        successful = [t for t in self.last_traj if t["success"]]
        n = max(len(successful), 1)

        dJ_deps_list = []
        for traj in self.last_traj:
            dJ_deps_traj = np.zeros(traj['state_traj_opt'].shape)
            if traj["success"]:
                eps_terminal = traj['state_traj_opt'][-1]
                # TODO: this works only because the target is (0,0..)
                # ocp.py divides by the successful count; do not divide again here
                dJ_deps_traj[-1, :] = 2 * (w * eps_terminal)

            dJ_deps_list.append(dJ_deps_traj)

        return dJ_deps_list
        
    def backward_time(self):
        successful = [t for t in self.last_traj if t["success"]]
        n = max(len(successful), 1)

        dJ_deps_list = []
        for traj in self.last_traj:
            dJ_deps_traj = np.zeros(traj['state_traj_opt'].shape)
            if traj["success"]:
                dJ_deps_traj[:, 7] = 1.0
            dJ_deps_list.append(dJ_deps_traj)

        return dJ_deps_list
    
    def _log_to_wandb(self, total_obj, cost_val, aug, iteration):
        metrics = {
            "evaluation/objective_total": total_obj,
            "evaluation/ocp_cost": cost_val,
            "evaluation/augmented_lagrangian": aug,
            **self._extra_metrics,
        }
        wandb.log(metrics, step=iteration)

    def plot_objective(self):
        out_dir = self._get_output_directory()
        run_name = self._get_run_name()
        
        self._save_plot(
            self.objective_evolution,
            "Total Objective",
            "Total Optimization Progress",
            out_dir / f"{run_name}_objective_total.png"
        )
        
        self._save_plot(
            self.cost_evolution,
            "Cost (OCP)",
            "OCP Cost Progress",
            out_dir / f"{run_name}_objective_cost.png"
        )
        
        return out_dir / f"{run_name}_objective_total.png", out_dir / f"{run_name}_objective_cost.png"

    def _get_output_directory(self):
        out_dir = Path(self.config.io.checkpoint_dir) if hasattr(self.config, "io") else Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _get_run_name(self):
        return getattr(self.config.io, "run_name", "run") if hasattr(self.config, "io") else "run"

    def _save_plot(self, data, ylabel, title, filepath):
        plt.figure()
        plt.plot(data)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        self.logger.info(f"Saved plot to {filepath}")