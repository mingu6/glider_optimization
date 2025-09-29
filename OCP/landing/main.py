from glider import Glider, State
from renderer import PygameRenderer
import numpy as np

max_t = 3
t = 0
dt = 0.003

glider = Glider()
glider.set_state(State(
    x = 0.0,
    z = 2.0,
    theta = 0.0,
    phi = 0.0,
    xdot = 1.0,
    zdot = 0.0,
    thetadot= 0.0
))

states = []

while t < max_t:
    state_dot = glider.update()

    next_state = glider.get_state() + state_dot * dt
    states.append(next_state.to_array())

    glider.set_state(next_state)

    t+=dt

def glider_state_parser(state):
        """
        Convert a generic state into drawable segments.
        Returns a list of segments (each segment is a list of points).
        Currently supports just the fuselage.
        """
        x, z, theta = state[:3] 
        half_length = 1.0
        points_body = np.array([[-half_length, 0.0], [half_length, 0.0]])
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        world_pts = (R @ points_body.T).T + np.array([x, z])
        return [world_pts.tolist()]

render = PygameRenderer(states,glider_state_parser)
render.run()