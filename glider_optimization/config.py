from pathlib import Path
import yaml
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np

class RunConfig(BaseModel):
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

class ArmConfig(BaseModel):
    """Throwing-arm co-design (see utils/arm_jinenv.py for the meaning of psi).

    The design vector psi is 13 numbers: 3 segment lengths, 3 tapers, 3 logits
    splitting a FIXED structural mass budget and 4 logits splitting a FIXED
    torque budget. Everything here that is not `lr`/`gamma`/bounds is a constant
    of the physical problem, not something the optimiser is allowed to move.
    """
    # outer-loop optimiser over psi
    lr: float = 5e-3
    gamma: float = 0.99
    initial_psi: list[float] = Field(
        default_factory=lambda: [0.32, 0.27, 0.11,
                                 1.0, 1.0, 1.0,
                                 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0]
    )
    length_min: float = 0.08
    length_max: float = 0.60
    taper_min: float = 0.35
    taper_max: float = 1.60

    # fixed budgets / physical constants
    shoulder_h: float = 1.30      # m, shoulder height above ground
    struct_mass: float = 3.60     # kg, total structural mass to distribute
    torque_budget: float = 170.0  # N m, total torque capacity to distribute
    motor_mass_per_Nm: float = 0.010
    ball_mass: float = 0.145      # kg
    proximal_radii: list[float] = Field(default_factory=lambda: [0.045, 0.038, 0.028])
    z_target: float = 1.00        # m, height of the target plane

    # inner OCP
    horizon: int = 40
    integrator: str = "rk4"
    # (x, y) landing points, one inner OCP each. A target the starting design
    # cannot comfortably reach is the point of the exercise: the residual miss
    # (and the swing time spent trying) is what drives the design. See
    # cold_start_stages for how the first solve gets there.
    targets: list[list[float]] = Field(default_factory=lambda: [[8.0, 0.0]])
    q_start: list[float] = Field(default_factory=lambda: [0.0, -2.10, 1.70, 0.55])
    # release pose used to shape the cold-start guess (azimuth is re-aimed per target)
    q_end_guess: list[float] = Field(default_factory=lambda: [0.0, 0.55, 0.30, -0.45])
    q_min: list[float] = Field(default_factory=lambda: [-2.6, -2.9, -0.2, -1.4])
    q_max: list[float] = Field(default_factory=lambda: [2.6, 1.2, 2.9, 1.4])
    h_min: float = 0.004          # s, timestep is a free variable of the OCP
    h_max: float = 0.020
    h_init: float = 0.010
    p_joint: float = 900.0        # W, peak mechanical power per joint

    # Cold-start continuation. Aiming straight at a hard target from the swing
    # guess lands IPOPT in a basin where it releases the ball downwards and
    # never recovers. Solving a near target first and walking the target out,
    # warm-starting each stage, converges every time. Only the first outer
    # iteration pays for it; after that the previous iteration's solution is the
    # warm start. Set stages to 1 to disable.
    # Distance of the first stage, in metres. Wants to be a mid-range throw for
    # the starting design: near throws and far throws are both harder than one
    # in the middle. The stages then have to be fine enough to cross the
    # design's max-range knee, where the solution changes character (timestep
    # pinned at h_max, throwing as far as it can rather than at the target).
    cold_start_first_distance: float = 4.5
    cold_start_stages: int = 9
    # intermediate stages are only seeds for the next one, so they get a smaller
    # iteration budget: a stage that is going nowhere costs seconds, not minutes
    cold_start_stage_max_iter: int = 600
    ipopt_options: dict[str, Any] = Field(
        default_factory=lambda: {"ipopt.mu_strategy": "adaptive",
                                 "ipopt.acceptable_tol": 1e-4,
                                 "ipopt.max_iter": 1500}
    )
    # N m/s, torque slew limit. A real motor cannot reverse its torque
    # instantly; without this the swing ambles and the throw happens in the
    # last one or two intervals. It is a bound on the control, since torque is
    # a state (see arm_jinenv.initDyn).
    du_max: float = 900.0
    w_miss: float = 400.0         # inner terminal weight on the squared miss
    w_time: float = 2.0           # inner stage weight on the timestep
    wu: float = 1e-5              # inner stage weight on torque effort
    w_taudot: float = 1e-8        # inner stage weight on torque RATE
    stage_scale: float = 1e-4     # inner stage weight on joint rates

    # outer objective (Evaluation block)
    outer_w_miss: float = 100.0
    outer_w_time: float = 1.0

    # throw animation on the first and last outer iteration
    animate: bool = True

    @field_validator("initial_psi")
    @classmethod
    def check_psi(cls, v):
        if len(v) != 13:
            raise ValueError(f"initial_psi must have exactly 13 elements, got {len(v)}")
        return v

    @field_validator("q_start", "q_end_guess", "q_min", "q_max")
    @classmethod
    def check_joint_vector(cls, v):
        if len(v) != 4:
            raise ValueError(f"joint vectors must have exactly 4 elements, got {len(v)}")
        return v

    @field_validator("proximal_radii")
    @classmethod
    def check_radii(cls, v):
        if len(v) != 3:
            raise ValueError(f"proximal_radii must have exactly 3 elements, got {len(v)}")
        return v

    @field_validator("targets")
    @classmethod
    def check_targets(cls, v):
        if not v:
            raise ValueError("at least one target is required")
        for t in v:
            if len(t) != 2:
                raise ValueError(f"each target must be an (x, y) pair, got {t}")
        return v

    @field_validator("integrator")
    @classmethod
    def check_integrator(cls, v):
        allowed = {"euler", "rk2", "rk4"}
        if v not in allowed:
            raise ValueError(f"integrator must be one of {allowed}, got '{v}'")
        return v

class NeuralFoilSamplingConfig(BaseModel):
    neuralFoil_size: str = "xxxlarge"
    AoA_min: float = -10.0
    AoA_max: float = 25.0
    Re_min: float = 1e4
    Re_max: float = 6e5
    n_samples: int = 100
    min_confidence: float = 0.7
    min_avg_Cl_Cd: float = 2.0
    rho: float = 10.0

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
    terminal_state_weight: list[float] = Field(
        default_factory = lambda: [10., 10., 5., 0.01, 5., 5., 2., 0.01]
    )
    stage_control_weight: float = 0.1
    initial_states: list[list[float]] = Field(
        default_factory= lambda : [[-8.5, 0 , 0. , 0., 6., 3. , 0., 0.01]]
    )
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
    RobotThrowing = "RobotThrowing"
class EvaluationConfig(BaseModel):
    mode: EvaluationMode = EvaluationMode.Perching
class Config(BaseModel):
    run: RunConfig
    airfoil: AirfoilConfig = Field(default_factory=AirfoilConfig)
    arm: ArmConfig = Field(default_factory=ArmConfig)
    neuralFoilSampling: NeuralFoilSamplingConfig = Field(default_factory=NeuralFoilSamplingConfig)
    reducedModel: ReducedModelConfig = Field(default_factory=ReducedModelConfig)
    io: IOConfig
    ocp: OCPConfig = Field(default_factory=OCPConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    plane: dict[str, Any] = Field(default_factory=dict)

def load_config(path: Path) -> Config:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
