"""
Utility functions for mathematical operations.
"""
import numpy as np

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to unit length (L2 norm).

    Args:
        v (np.ndarray): Feature vector

    Returns:
        np.ndarray: L2 normalized vector
    """
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n
