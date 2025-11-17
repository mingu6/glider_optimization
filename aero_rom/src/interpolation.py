# RGI 'cubic' training & persistence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

class RGISurface:
    """
    RegularGridInterpolator wrapper with method='cubic' and derivative access via nu.
    Assumes a tensor grid: alpha_grid (deg), velocity_grid (m/s), values shape (Na, Nv).
    """
    def __init__(self, alpha_grid_deg, velocity_grid, values_2d,
                 method="cubic", bounds_error=False, fill_value=None):
        self.alpha = np.asarray(alpha_grid_deg, float)
        self.vel   = np.asarray(velocity_grid, float)
        vals       = np.asarray(values_2d, float)
        assert vals.shape == (self.alpha.size, self.vel.size), \
            f"values must be (Na, Nv), got {vals.shape} vs {(self.alpha.size, self.vel.size)}"
        self._rgi = RegularGridInterpolator(
            (self.alpha, self.vel), vals,
            method=method, bounds_error=bounds_error, fill_value=fill_value
        )

    def __call__(self, alpha_deg, velocity):
        xi = np.atleast_2d([alpha_deg, velocity])
        out = self._rgi(xi)  
        return float(out[0])

    def grad(self, alpha_deg, velocity):
        xi = np.atleast_2d([alpha_deg, velocity])
        d_da = float(self._rgi(xi, nu=(1, 0))[0])
        d_dv = float(self._rgi(xi, nu=(0, 1))[0])
        return d_da, d_dv

    def to_npz(self, path):
        np.savez_compressed(path,
                            alpha=self.alpha,
                            vel=self.vel,
                            values=self._rgi.values)

    @staticmethod
    def from_npz(path, method="cubic", bounds_error=False, fill_value=None):
        d = np.load(path)
        return RGISurface(d["alpha"], d["vel"], d["values"],
                          method=method, bounds_error=bounds_error, fill_value=fill_value)
