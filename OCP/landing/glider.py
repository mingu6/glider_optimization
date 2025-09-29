import numpy as np
    
class Glider:
    def __init__(self, 
                 m=0.065, I=0.2, g=9.81, 
                 k=1e4, mu=0.2, c=None, half_length=1.0):
        self.m = m
        self.I = I
        self.g = g
        self.k = k
        self.mu = mu
        self.c = c if c is not None else 2*np.sqrt(k*m)
        self.half_length = half_length
        self.points_body = np.array([[-half_length, 0.0], [half_length, 0.0]])

        # state
        self.x = 0.0
        self.z = 2.0
        self.theta = 0.2
        self.vx = 3.0
        self.vz = 5.0
        self.omega = 0.0

    def R(self, theta):
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, -st],[st, ct]])

    def world_points(self):
        return (self.R(self.theta) @ self.points_body.T).T + np.array([self.x, self.z])

    def vel_point(self, p):
        r = p - np.array([self.x, self.z])
        return np.array([self.vx, self.vz]) + np.array([-self.omega * r[1], self.omega * r[0]])

    def step(self, dt=1e-3):
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
            r = p - np.array([self.x, self.z])
            tau += (r[0]*Fn - r[1]*Ft) * sigmoid(delta)

        # integrate
        self.vx += (F[0]/self.m) * dt
        self.vz += (F[1]/self.m) * dt
        self.omega += (tau/self.I) * dt
        self.x += self.vx * dt
        self.z += self.vz * dt
        self.theta += self.omega * dt

    def get_state(self):
        return np.array([self.x, self.z, self.theta, self.vx, self.vz, self.omega])
    
    def set_state(self, state):
        self.x, self.z, self.theta, self.vx, self.vz, self.omega = state
