from pathlib import Path
import yaml
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np

class RunConfig(BaseModel):
    seed: int = 0
    device: str = "cpu"
    max_outer_iters: int = 50

# SpanwiseAirfoilConfig enables optional root->tip airfoil interpolation for 3D LLT runs.
# If disabled, 2D behavior is unchanged.

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
        default_factory=lambda: np.array([0.18109497, 0.21268419, 0.28098503, 0.24864887, 0.2402814, 0.27262843, 0.25776474, 0.27817638])
    )
    lower_initial_weights: np.ndarray = Field(
        default_factory=lambda: np.array([-0.16965146, -0.09364138, -0.06345896, -0.0067966, -0.0902447,  0.02081845, -0.03575216, -0.00223623])
    )
    leading_edge_weight: float = 0.10647
    TE_thickness: float = 0.00257
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
    
    # model_validator is used for cross-field checks (e.g. requiring plane geometry in 3D LLT mode)


    @model_validator(mode="after")
    # If spanwise airfoil interpolation is enabled but tip parameters are omitted,
    # we copy root parameters to tip so the 3D pipeline always has valid root+tip definitions.
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
    sigma: float = 10.0 #previously rho, but renamed to avoid confusion with flow density rho in FlowConfig

    # Optional: upgrade 2D sampling to 3D LLT
    use_3d_llt: bool = False

    # Path to aero_rom checkpoint (contains wing geometry + flow used to invert Re -> V)
    llt_ckpt_path: str = "artifacts/models/3d_blocks.pt"

    # Optional overrides (if None, fall back to checkpoint values)
    llt_n_iter: int | None = None
    llt_beta: float | None = None
    llt_tol: float | None = None
    llt_enforce_symmetry: bool | None = None

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
    rho: float = 1.225          # density of air at sea level in kg/m^3
    mu: float = 1.789e-5        # dynamic viscosity of air at sea level in kg/(m*s)

class SurfaceGeometryConfig(BaseModel):
    y_half: list[float]         # spanwise y-coordinates of the half-chord points (for 3D LLT, these define the spanwise sampling locations of the surrogate model blocks)
    c_half: list[float]         # spanwise chord lengths at the half-chord points (for 3D LLT, these are used to non-dimensionalize the surrogate model inputs/outputs)
    xle_half: list[float]       # spanwise x-coordinates of the leading edge points (for 3D LLT, these define the spanwise leading edge locations of the surrogate model blocks)
    twist_half: list[float]     # spanwise twist angles at the half-chord points (for 3D LLT, these define the spanwise twist distribution of the surrogate model blocks)   
    x_ref: float = 0.0          # x-coordinate of reference point for moment calculations (e.g. aerodynamic center, quarter-chord, etc.)
    z_ref: float = 0.0          # z-coordinate of reference point for moment calculations (e.g. aerodynamic center, quarter-chord, etc.)

    # Backward compatible single airfoil name
    airfoil: str | None = None

    # 3D-only spanwise wing airfoil names (optional)
    airfoil_root: str | None = None
    airfoil_tip: str | None = None

    use_quarter_chord_ref: bool = True

class DynConfig(BaseModel):
    mass: float = 0.065     # mass of the glider in kg
    S_w: float = 0.158      # wing area in m^2
    S_e: float = 0.017      # elevator area in m^2
    l_w_i: float = -0.005   # l_w_i is distance from center of mass to wing aerodynamic center (positive if AC is in front of CoM, negative if behind)
    l_e_i: float = 0.01     # l_e_i is distance from center of mass to elevator aerodynamic center (positive if AC is in front of CoM, negative if behind)
    l_w_f: float = -0.015   # l_w_f is distance from center of mass to point where wing forces are applied in the surrogate model (for 3D LLT, this is typically the spanwise location of the surrogate block, e.g. y_half)
    l: float = 0.26         # reference length for non-dimensionalization (e.g. mean aerodynamic chord)
    l_e: float = 0.02       # l_e is reference length for elevator non-dimensionalization (for 3D LLT, this is typically the chord length at the surrogate block location, e.g. c_half)

# PlaneConfig groups all physical definitions required for 3D LLT:
# flow properties (rho, mu) + wing/elevator geometry + dynamics constants.
# Not used in 2D-only mode to keep legacy behavior unchanged.
class PlaneConfig(BaseModel):
    flow: FlowConfig                # flow properties (e.g. density, viscosity) used for 3D LLT surrogate model sampling and Re->V inversion
    wing: SurfaceGeometryConfig     # wing geometry definition for 3D LLT surrogate model (spanwise y_half, c_half, xle_half, twist_half, etc.)
    elevator: SurfaceGeometryConfig | None = None     # optional elevator geometry definition for 3D LLT surrogate model (if None, elevator surrogate blocks will not be used)
    dyn: DynConfig                  # dynamics constants (mass, reference areas, moment arms, etc.) used for 3D LLT surrogate model sampling and OCP formulation    

class Config(BaseModel):
    run: RunConfig
    airfoil: AirfoilConfig = Field(default_factory=AirfoilConfig) 
    neuralFoilSampling: NeuralFoilSamplingConfig = Field(default_factory=NeuralFoilSamplingConfig)
    reducedModel: ReducedModelConfig = Field(default_factory=ReducedModelConfig)
    plane: PlaneConfig | None = None
    io: IOConfig
    ocp: OCPConfig 
    # plane is required only for 3D LLT runs; in 2D it is None to preserve legacy behavior.
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
