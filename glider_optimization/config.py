from pathlib import Path
import yaml
from typing import Any
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
    rho: float = 10.0

    # Optional: upgrade 2D sampling to 3D LLT
    use_3d_llt: bool = False

    # Path to aero_rom checkpoint (contains wing geometry + flow used to invert Re -> V)
    llt_ckpt_path: str = "aero_rom/artifacts/models/3d_blocks.pt"

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

class IOConfig(BaseModel):
    gif_fps: int = 1
    log_every: int = 1
    checkpoint_dir: str
    metrics: list[str] = Field(default_factory=list)
    run_name: str = "run"
    debug: bool = False
    
class Config(BaseModel):
    run: RunConfig
    airfoil: AirfoilConfig = Field(default_factory=AirfoilConfig) 
    neuralFoilSampling: NeuralFoilSamplingConfig = Field(default_factory=NeuralFoilSamplingConfig)
    reducedModel: ReducedModelConfig = Field(default_factory=ReducedModelConfig)
    io: IOConfig

def load_config(path: Path) -> Config:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
