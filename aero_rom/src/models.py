# ClModel, CdModel, CmModel (__call__ + backward)

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from src.interpolation import RGISurface

@dataclass
class _BaseCoeff:
    surface: RGISurface
    clip_alpha: Optional[Tuple[float, float]] = None
    clip_vel:   Optional[Tuple[float, float]] = None

    def _clip(self, a, v):
        if self.clip_alpha is not None:
            a = min(max(a, self.clip_alpha[0]), self.clip_alpha[1])
        if self.clip_vel is not None:
            v = min(max(v, self.clip_vel[0]), self.clip_vel[1])
        return a, v

    def __call__(self, alpha_deg: float, velocity: float) -> float:
        a, v = self._clip(alpha_deg, velocity)
        return self.surface(a, v)

    def backward(self, alpha_deg: float, velocity: float):
        """
        Returns tuple: (dc/dalpha [per degree], dc/dV [per m/s])
        """
        a, v = self._clip(alpha_deg, velocity)
        return self.surface.grad(a, v)

class ClModel(_BaseCoeff): 
    pass
class CdModel(_BaseCoeff): 
    pass
class CmModel(_BaseCoeff): 
    pass
