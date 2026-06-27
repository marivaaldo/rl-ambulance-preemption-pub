"""
Global seed utility — call once at the top of each script that uses randomness.

Fixes `random`, `numpy`, `torch`, and `torch.cuda` to the given seed.
Note: the SUMO/TraCI simulation is deterministic given a fixed route file;
for full reproducibility also pass `seed` to `env.reset(seed=...)`.
"""
import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[seed] Global seed set to {seed}")
