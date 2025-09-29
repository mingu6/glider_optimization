import numpy as np
from dataclasses import dataclass
@dataclass
class State:
    x: float
    z: float
    theta: float
    phi: float
    xdot: float
    zdot: float
    thetadot: float
    
    def __mul__(self, scalar: float):
        return State(
            self.x * scalar,
            self.z * scalar,
            self.theta * scalar,
            self.phi * scalar,
            self.xdot * scalar,
            self.zdot * scalar,
            self.thetadot * scalar
        )

    __rmul__ = __mul__

    def __add__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return State(
            self.x + other.x,
            self.z + other.z,
            self.theta + other.theta,
            self.phi + other.phi,
            self.xdot + other.xdot,
            self.zdot + other.zdot,
            self.thetadot + other.thetadot
        )
    
    __radd__ = __add__

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.z, self.theta, self.phi, self.xdot, self.zdot, self.thetadot])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(*arr)

class BodyPart:
    def __init__(self, name, points, mass, I, parent=None, joint_idx=None):
        self.name = name
        self.points = np.array(points)   # Nx2 points in local frame
        self.mass = mass
        self.I = I
        self.parent = parent             # None for main body
        self.joint_idx = joint_idx       # index in State for rotation

class Glider:
    def __init__(self, 
                 m=0.065, I=0.2, g=9.81, 
                 k=1e3, mu=0.2, c=None, half_length=1.0):
        self.m = m
        self.I = I
        self.g = g
        self.k = k
        self.mu = mu
        self.c = c if c is not None else 2*np.sqrt(k*m)
        self.half_length = half_length
        self.points_body = np.array([[-half_length, 0.0], [half_length, 0.0]])

        # state
        self.state = State(
            x = 0.0,
            z = 2.0,
            theta = 0.2,
            phi = 0.0,
            xdot = 0.0,
            zdot = 0.0,
            thetadot= 0.0
        )

        # control 
        self.phidot = 0.0

    def R(self, theta):
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, -st],[st, ct]])

    def world_points(self):
        return (self.R(self.state.theta) @ self.points_body.T).T + np.array([self.state.x, self.state.z])

    def vel_point(self, p):
        r = p - np.array([self.state.x, self.state.z])
        return np.array([self.state.xdot, self.state.zdot]) + np.array([-self.state.thetadot * r[1], self.state.thetadot * r[0]])

    def update(self):
        F = np.array([0.0, -self.m*self.g])
        tau = 0.0
        pts = self.world_points()
        for p in pts:
            delta = -p[1]
            vcp = self.vel_point(p)
            Fn = self.k * delta - self.c * vcp[1]
            Fn = max(Fn, 0.0)
            Ft = -self.mu * Fn * np.sign(vcp[0]) if Fn > 0 else 0.0

            sigmoid = lambda x: 1/(1+np.exp(-400*x))
            F += np.array([Ft, Fn]) * sigmoid(delta)
            r = p - np.array([self.state.x, self.state.z])
            tau += (r[0]*Fn - r[1]*Ft) * sigmoid(delta)
        xdotdot = F[0]/self.m
        zdotdot = F[1]/self.m
        thetadotdot = tau/self.I

        return State(
            x = self.state.xdot,
            z = self.state.zdot,
            theta = self.state.thetadot,
            phi = self.phidot,
            xdot = xdotdot,
            zdot = zdotdot,
            thetadot = thetadotdot
        )
    
    def get_state(self):
        return self.state
    
    def get_control(self):
        return self.phidot
    
    def set_control(self, u):
        self.phidot = u
    
    def set_state(self, state: State):
        self.state = state
