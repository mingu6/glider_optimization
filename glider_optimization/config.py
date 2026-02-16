from pathlib import Path
import yaml
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np

class RunConfig(BaseModel):
    seed: int = 0
    device: str = "cpu"
    max_outer_iters: int = 50

class SpanwiseAirfoilConfig(BaseModel):
    """
    Optional root->tip airfoil interpolation (used by 3D LLT only).
    Tip fields default to root values if omitted (handled in Airfoil block too).
    """
    enabled: bool = False
    tip_upper_initial_weights: Optional[np.ndarray] = None
    tip_lower_initial_weights: Optional[np.ndarray] = None
    tip_leading_edge_weight: Optional[float] = None
    tip_TE_thickness: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("tip_upper_initial_weights", "tip_lower_initial_weights", mode="before")
    @classmethod
    def validate_tip_arrays(cls, v: Any) -> Optional[np.ndarray]:
        if v is None:
            return None
        arr = np.array(v, dtype=float)
        if arr.shape[0] != 8:
            raise ValueError(f"{arr} must have exactly 8 elements")
        return arr

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
    spanwise: SpanwiseAirfoilConfig = Field(default_factory=SpanwiseAirfoilConfig)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("upper_initial_weights", "lower_initial_weights", mode="before")
    @classmethod
    def validate_array(cls, v: Any) -> np.ndarray:
        arr = np.array(v, dtype=float)
        if arr.shape[0] != 8:
            raise ValueError(f"{arr} must have exactly 8 elements")
        return arr
    
    @model_validator(mode="after")
    def fill_spanwise_defaults(self):
        sp = getattr(self, "spanwise", None)
        if sp is None or not sp.enabled:
            return self

        if sp.tip_upper_initial_weights is None:
            sp.tip_upper_initial_weights = np.array(self.upper_initial_weights, dtype=float)
        if sp.tip_lower_initial_weights is None:
            sp.tip_lower_initial_weights = np.array(self.lower_initial_weights, dtype=float)
        if sp.tip_leading_edge_weight is None:
            sp.tip_leading_edge_weight = float(self.leading_edge_weight)
        if sp.tip_TE_thickness is None:
            sp.tip_TE_thickness = float(self.TE_thickness)

        # Make sure types are right
        sp.tip_upper_initial_weights = np.array(sp.tip_upper_initial_weights, dtype=float)
        sp.tip_lower_initial_weights = np.array(sp.tip_lower_initial_weights, dtype=float)

        self.spanwise = sp
        return self

class NeuralFoilSamplingConfig(BaseModel):
    neuralFoil_size: str = "xxxlarge"
    AoA_min: float = -10.0
    AoA_max: float = 25.0
    Re_min: float = 1e4
    Re_max: float = 6e5
    n_samples: int = 100
    min_confidence: float = 0.7
    min_avg_Cl_Cd: float = 2.0
    sigma: float = 10.0

    # Optional: upgrade 2D sampling to 3D LLT
    use_3d_llt: bool = False

    # Path to aero_rom checkpoint (contains wing geometry + flow used to invert Re -> V)
    llt_ckpt_path: str = "aero_rom/artifacts/models/3d_blocks.pt"

    # Optional overrides (if None, fall back to checkpoint values)
    llt_n_iter: int | None = None
    llt_max_iter: int | None = None  # Maximum iterations for adaptive convergence
    llt_beta: float | None = None
    llt_tol: float | None = None
    llt_enforce_symmetry: bool | None = None
    
    # Backward pass mode: "implicit" (fast), "explicit" (robust), or "hybrid" (auto-fallback)
    llt_backward_mode: str = "hybrid"
    llt_fallback_residual_threshold: float = 1e-4  # Residual threshold to trigger explicit fallback

    # cuNF size used inside LLT (defaults to neuralFoil_size)
    llt_model_size: str | None = None
    
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

class OCPConfig(BaseModel):
    terminal_state_weight: list[float] = Field(
        default_factory = lambda: [10., 10., 5., 0.01, 5., 5., 2., 0.01]
    )
    
    stage_control_weight: float = 0.1
    
    initial_state: list[float] = Field(
        default_factory= lambda : [-8.5, 0 , 0. , 0., 6., 3. , 0., 0.01]
    )

    # Backward-compatibility: allow other YAML schemas (e.g. n_initial_conditions,
    # init_state_ranges, etc.) without breaking config parsing.
    # If the YAML has additional keys under ocp: that are not explicitly declared in OCPConfig,Pydantic will not error; it will accept them.

    model_config = {"extra": "allow"} 
    
class IOConfig(BaseModel):
    gif_fps: int = 1
    log_every: int = 1
    checkpoint_dir: str
    metrics: list[str] = Field(default_factory=list)
    run_name: str = "run"
    debug: bool = False
    wandb: WandbConfig = Field(default_factory=WandbConfig)

class FlowConfig(BaseModel):
    rho: float = 1.225
    mu: float = 1.789e-5

class SurfaceGeometryConfig(BaseModel):
    y_half: list[float]
    c_half: list[float]
    xle_half: list[float]
    twist_half: list[float]
    x_ref: float = 0.0
    z_ref: float = 0.0

    # Backward compatible single airfoil name
    airfoil: str | None = None

    # 3D-only spanwise wing airfoil names (optional)
    airfoil_root: str | None = None
    airfoil_tip: str | None = None

    use_quarter_chord_ref: bool = True

class DynConfig(BaseModel):
    mass: float = 0.065
    S_w: float = 0.158
    S_e: float = 0.017
    l_w_i: float = -0.005
    l_w_f: float = -0.015
    l: float = 0.26
    l_e: float = 0.02

class PlaneConfig(BaseModel):
    flow: FlowConfig
    wing: SurfaceGeometryConfig
    elevator: SurfaceGeometryConfig | None = None     # keep placeholder
    dyn: DynConfig

class Config(BaseModel):
    run: RunConfig
    airfoil: AirfoilConfig = Field(default_factory=AirfoilConfig) 
    neuralFoilSampling: NeuralFoilSamplingConfig = Field(default_factory=NeuralFoilSamplingConfig)
    reducedModel: ReducedModelConfig = Field(default_factory=ReducedModelConfig)
    plane: PlaneConfig | None = None
    io: IOConfig
    ocp: OCPConfig 
    @model_validator(mode="after")
    def plane_only_for_3d(self):
        if self.neuralFoilSampling.use_3d_llt:
            if self.plane is None:
                raise ValueError("plane: must be provided when use_3d_llt: true")
        else:
            # keep old behavior as default baseline
            self.plane = None
        return self
    
def load_config(path: Path) -> Config:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
