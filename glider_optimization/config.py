from pathlib import Path
import yaml
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np

class RunConfig(BaseModel):
    seed: int = 0
    device: str = "cpu"
    max_outer_iters: int = 50
    cost_target: float | None = None
    cost_target_min_iters: int = 0
    cost_residual_tol: float | None = None
    cost_residual_patience: int = 3
    cost_residual_min_iters: int = 5
    
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
    # Optional: separate initial parameters for tip (3D only).
    upper_initial_weights_tip: np.ndarray | None = None
    lower_initial_weights_tip: np.ndarray | None = None
    leading_edge_weight_tip: float | None = None
    TE_thickness_tip: float | None = None
    N1: float = 0.5
    N2: float = 1.0
    gamma: float = 0.99

    model_config = {"arbitrary_types_allowed": True}

    @field_validator(
        "upper_initial_weights", "lower_initial_weights",
        "upper_initial_weights_tip", "lower_initial_weights_tip",
        mode="before"
    )
    @classmethod
    def validate_array(cls, v: Any) -> np.ndarray | None:
        if v is None:
            return None
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
    min_avg_Cl_Cd: float = 2.0 #2.5
    rho: float = 1.0

    # Optional: upgrade 2D sampling to 3D LLT
    use_3d_llt: bool = False

    # Sugar-Gabor (2018) quasi-unsteady corrections in OCP dynamics (Terms 2 & 3, Eq. 13)
    unsteady: bool = False

    # Solver settings
    llt_n_iter: int = 30
    llt_max_iter: int = 200
    llt_beta: float = 0.5
    llt_tol: float = 1e-5

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
    stage_control_weight: float = 0.02 #0.1

    # Backward-compatible fallback used when mode-specific lists are not provided
    initial_states: list[list[float]] = Field(
        default_factory= lambda : [[-8.5, 0 , 0. , 0., 6., 3. , 0., 0.01]]
    )

    # Preferred mode-specific initial conditions
    initial_states_perching: list[list[float]] = Field(
        default_factory=lambda: [[-8.5, 0.0, 0.0, 0.0, 6.0, 3.0, 0.0, 0.01]]
    )
    initial_states_softlanding: list[list[float]] = Field(
        default_factory=lambda: [[-8.5, 1.0, 0.0, 0.0, 2.5, -1.0, 0.0, 0.01]]
    )
class IOConfig(BaseModel):
    gif_fps: int = 1
    log_every: int = 1
    static_plot_every: int = 5
    airfoil_gif_every: int = 5
    checkpoint_dir: str
    metrics: list[str] = Field(default_factory=list)
    run_name: str = "run"
    debug: bool = False
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    
class EvaluationMode(str, Enum):
    Perching = "Perching"
    Time = "Time"
    SoftLanding = "SoftLanding"
class EvaluationConfig(BaseModel):
    mode: EvaluationMode = EvaluationMode.Perching
class Config(BaseModel):
    run: RunConfig
    airfoil: AirfoilConfig = Field(default_factory=AirfoilConfig) 
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
