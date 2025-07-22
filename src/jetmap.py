import numpy as np
from typing import Tuple

def jetmap(v: float, vmin: float, vmax: float) -> np.ndarray:
    "convert scalar value to RGB using jet colormap"

    c = np.ones(3)
    if np.isclose(vmin, vmax):
        return np.array([0.5, 0.5, 0.5])  # fallback color for flat data

    v = np.clip(v, vmin, vmax)
    dv = vmax - vmin

    if v < (vmin + 0.25 * dv):
        c[0] = 0
        c[1] = 4 * (v - vmin) / dv
        c[2] = 1
    elif v < (vmin + 0.5 * dv):
        c[0] = 0
        c[1] = 1
        c[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv
    elif v < (vmin + 0.75 * dv):
        c[0] = 4 * (v - vmin - 0.5 * dv) / dv
        c[1] = 1
        c[2] = 0
    else:
        c[0] = 1
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        c[2] = 0

    return np.clip(c, 0, 1)
