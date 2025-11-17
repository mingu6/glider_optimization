# CSV/NPZ read/write (surfaces & models)

import numpy as np
import pandas as pd

def save_surface_csv(path, alpha, vel, values, value_name):
    df = pd.DataFrame(values, index=alpha, columns=vel)
    df.index.name = "alpha_deg"
    df.columns.name = "velocity"
    df.to_csv(path, float_format="%.8g")

def load_surface_csv(path):
    df = pd.read_csv(path, index_col=0)
    alpha = df.index.values.astype(float)
    vel = df.columns.values.astype(float)
    values = df.values.astype(float)
    return alpha, vel, values
