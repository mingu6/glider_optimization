# load/validate JSON config

import json
from pathlib import Path
import numpy as np

def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    # optionally validate keys/units; keep strict but simple
    return cfg

