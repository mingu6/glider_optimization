"""Utilities for interacting with Weights & Biases."""

import wandb
import numpy as np
import logging
from typing import Dict, Optional


def load_checkpoint_from_wandb(
    run_id: str,
    iteration: int,
    project: str,
    entity: Optional[str] = None
) -> Dict[str, np.ndarray]:
    logger = logging.getLogger(__name__)
    
    api = wandb.Api()
    
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"
    
    logger.info(f"Fetching airfoil parameters from wandb run: {run_path}, iteration: {iteration}")
    
    run = api.run(run_path)
    
    history = run.history()
    
    matching_rows = history[history['_step'] == iteration]
    
    if matching_rows.empty:
        raise ValueError(
            f"No data found for iteration {iteration} in run {run_id}. "
            f"Available steps: {sorted(history['_step'].unique())}"
        )
    
    row = matching_rows.iloc[0]
    return row
    
    