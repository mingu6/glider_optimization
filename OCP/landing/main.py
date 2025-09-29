from glider import Glider
from renderer import PygameRenderer
import numpy as np

max_t = 3
t = 0

glider = Glider()

states = []

while t < max_t:
    glider.step()
    states.append(glider.get_state())
    t+=0.001

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