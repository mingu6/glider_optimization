from pathlib import Path
import yaml
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np

class RunConfig(BaseModel):

    # Optional file holding Adam moments, the LR schedule position and the
    # augmented-Lagrangian duals. Continuation links are adjacent problems --
    # lambda for the epsilon constraint is nearly the same between neighbouring
    # bounds -- so restarting them at lambda=0, rho=rho0 with fresh Adam moments
    # makes every link re-learn what the previous one already knew.
    chain_state_path: Optional[str] = None
    seed: int = 0
    device: str = "cpu"
    max_outer_iters: int = 50
    
class AirfoilConfig(BaseModel):
    lr: float = 1e-2
    upper_initial_weights: np.ndarray = Field(
        default_factory=lambda: np.array([0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01])
    )
    lower_initial_weights: np.ndarray = Field(
        default_factory=lambda: np.array([-0.05, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005, 0.0])
    )
    leading_edge_weight: float = 0.0
    TE_thickness: float = 0.0
    N1: float = 0.5
    N2: float = 1.0
    gamma: float = 0.99
    model_config = {"arbitrary_types_allowed": True}
    

    @field_validator("upper_initial_weights", "lower_initial_weights", mode="before")
    @classmethod
    def validate_array(cls, v: Any) -> np.ndarray:
        arr = np.array(v, dtype=float)
        if arr.shape[0] != 8:
            raise ValueError(f"{arr} must have exactly 8 elements")
        return arr

class NeuralFoilSamplingConfig(BaseModel):
    neuralFoil_size: str = "xxxlarge"
    AoA_min: float = -10.0
    AoA_max: float = 25.0
    Re_min: float = 1e4
    Re_max: float = 6e5
    n_samples: int = 100
    min_confidence: float = 0.7
    min_avg_Cl_Cd: float = 2.0

    # Augmented Lagrangian: initial penalty, its growth rule and the dual cadence.
    rho: float = 10.0
    rho_max: float = 1e4
    rho_factor: float = 2.0

    # Penalty weight for the *task* (epsilon) constraints, separate from the design
    # constraints above. The landing-cost design gradient measures ~3.6x weaker than
    # the true slope, so at the shared rho the bound cannot hold the design against
    # the glide objective and the margin never closes. None keeps them equal.
    task_rho: Optional[float] = None
    task_rho_max: Optional[float] = None
    infeasibility_decrease: float = 0.75
    penalty_every: int = 10
    dual_every: int = 5

    use_3d_llt: bool = False
    llt_beta: float = 0.5
    llt_tol: float = 1e-5
    llt_max_iter: int = 200

    @field_validator("neuralFoil_size")
    def check_neuralFoil_size(cls, v):
        allowed = {"xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"}
        if v not in allowed:
            raise ValueError(f"neuralFoil_size must be one of {allowed}, got '{v}'")
        return v
    
class ReducedModelConfig(BaseModel):
    chebyshev_degree: int = 17
    l2_reg: float = 0.5

class WandbConfig(BaseModel):
    enabled: bool = True
    project: str = "glider-optimization"
    entity: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    
    checkpoint_run_id: Optional[str] = None
    checkpoint_iteration: Optional[int] = None

class OCPConfig(BaseModel):
    # Number of transcription nodes. The timestep is a decision variable, so this
    # sets the resolution the flight is resolved at: dt ~ flight_time / horizon.
    # Tasks whose optimal flight lengthens as the design improves need enough nodes
    # to keep dt small against the glider's stiff pitch mode, or RK4 stops
    # representing it and the NLP becomes ill-conditioned.
    horizon: int = 111
    terminal_state_weight: list[float] = Field(
        default_factory = lambda: [10., 10., 5., 0.01, 5., 5., 2., 0.01]
    )
    stage_control_weight: float = 0.1
    initial_states: list[list[float]] = Field(
        default_factory= lambda : [[-8.5, 0 , 0. , 0., 6., 3. , 0., 0.01]]
    )

    # EvaluationMode.Distance only: reward on horizontal distance covered, and the
    # terminal-altitude weight that ends the free-final-time flight on the ground.
    distance_weight: float = 1.0
    touchdown_weight: float = 50.0

    # Stage penalty on wing angle of attack (rad^2 per node), 0 disables it.
    aoa_weight: float = 0.0

    # SoftLanding only: hard bound on |alpha_w| in degrees, keeping the trajectory
    # inside the range where the NeuralFoil surrogate (and hence the airfoil) governs.
    aoa_max: float = 25.0

    # SoftTouchdown: weight on terminal kinetic energy in the OCP cost. The
    # objective itself is that energy in joules; this is what makes the trajectory
    # chase it rather than merely report it.
    touchdown_energy_weight: float = 1.0

    # Glider mass (kg), used to report touchdown energy in joules.
    glider_mass: float = 0.065

    # Distance/ShortLanding: terminal penalty on sink rate. ShortLanding's objective
    # is distance in metres, so without this the shortest landing is simply the
    # hardest one. Stiff enough that |zdot_T| stays well under ~0.5 m/s.
    sink_weight: float = 200.0

    # Reject a solve whose specific energy z + v^2/2g rises by more than this (m).
    # These tasks are unpowered, so energy can only decrease; a warm start can put
    # IPOPT somewhere it terminates at a dynamically infeasible point and still
    # reports Solve_Succeeded. One such trajectory climbed from 5.8 m to 242 m of
    # specific energy and flew 174 m, inflating the mean objective for 26 iterations
    # before it was noticed. Set to 0 or below to disable the check.
    energy_gain_tol: float = 0.05

    # Optional pickle holding the warm-start reservoir. Loaded at start-up and
    # rewritten each iteration, so successive processes (e.g. the links of a
    # Pareto continuation) continue from warm solves rather than cold ones.
    warm_start_path: Optional[str] = None
class IOConfig(BaseModel):
    gif_fps: int = 1
    log_every: int = 1
    checkpoint_dir: str
    metrics: list[str] = Field(default_factory=list)
    run_name: str = "run"
    debug: bool = False
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    
class EvaluationMode(str, Enum):
    Perching = "Perching"
    Time = "Time"
    SoftLanding = "SoftLanding"
    Distance = "Distance"
    ShortLanding = "ShortLanding"
    SoftTouchdown = "SoftTouchdown"
class EvaluationConfig(BaseModel):
    mode: EvaluationMode = EvaluationMode.Perching
class TaskConfig(BaseModel):
    """One task in the multi-task co-design problem.

    The first task in ``Config.tasks`` is the primary objective Phi_1; every
    later one is a *requirement*, entering as the epsilon-constraint
    ``Phi_j(psi) <= epsilon``.  Each bound is stated in its own task's units, so
    nothing has to be normalised against anything else, and sweeping epsilon
    traces the Pareto front including any non-convex stretches that a weighted
    sum of the task costs could never reach.
    """

    name: str
    mode: EvaluationMode
    initial_states: list[list[float]]
    epsilon: Optional[float] = None      # required for every task after the first
    distance_weight: float = 1.0
    touchdown_weight: float = 50.0

    # Treat this task's epsilon bound as an equality (Phi_j == eps) rather than an
    # inequality (Phi_j <= eps). At the optimum of an epsilon-subproblem the bound is
    # active anyway; declaring it an equality drives the margin to zero directly
    # instead of waiting for the multiplier to oscillate onto the boundary, which in
    # practice left margins of +6 to +14 after 40 iterations.
    epsilon_active: bool = False

    # Control-effort weight for this task; falls back to OCPConfig.stage_control_weight.
    # Distance/ShortLanding scale their stage cost by dt (a time integral) while the
    # quadratic modes sum per node, so the two want values an order of magnitude apart.
    stage_control_weight: Optional[float] = None

    # Transcription nodes for this task; falls back to OCPConfig.horizon. Tasks whose
    # flights differ in duration should not share a node count: a ~2 s landing at the
    # ~10 s distance task's 500 nodes is resolved at dt = 0.004 s, paying for accuracy
    # nobody asked for while every iteration waits on it.
    horizon: Optional[int] = None

    @field_validator("initial_states")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("a task needs at least one initial state")
        return v


class Config(BaseModel):
    run: RunConfig
    airfoil: AirfoilConfig = Field(default_factory=AirfoilConfig) 
    neuralFoilSampling: NeuralFoilSamplingConfig = Field(default_factory=NeuralFoilSamplingConfig)
    reducedModel: ReducedModelConfig = Field(default_factory=ReducedModelConfig)
    io: IOConfig
    ocp: OCPConfig = Field(default_factory=OCPConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    plane: dict[str, Any] = Field(default_factory=dict)

    # Multi-task (epsilon-constraint) co-design. Empty means single-task, driven by
    # `evaluation.mode` and `ocp.initial_states` as before.
    tasks: list[TaskConfig] = Field(default_factory=list)

    @field_validator("tasks")
    @classmethod
    def secondary_tasks_need_epsilon(cls, v):
        for task in v[1:]:
            if task.epsilon is None:
                raise ValueError(f"task '{task.name}' is a requirement and needs an epsilon bound")
        if v and v[0].epsilon is not None:
            raise ValueError(f"task '{v[0].name}' is the primary objective and must not have an epsilon")
        names = [t.name for t in v]
        if len(set(names)) != len(names):
            raise ValueError(f"task names must be unique, got {names}")
        return v

    @property
    def task_list(self) -> list[TaskConfig]:
        """Tasks as configured, or the single implied task for legacy configs."""
        if self.tasks:
            return self.tasks
        return [TaskConfig(name=self.evaluation.mode.value.lower(), mode=self.evaluation.mode,
                           initial_states=self.ocp.initial_states,
                           distance_weight=self.ocp.distance_weight,
                           touchdown_weight=self.ocp.touchdown_weight)]

def load_config(path: Path) -> Config:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
