"""
Utility functions for geometric calculations with points.
"""

import numpy as np


def distance_to_point(point_A: np.ndarray, point_B: np.ndarray) -> float:
    """Calculate the Euclidean distance of a point to another point.

    Args:
        point_A (numpy.ndarray): [x, y, z]
        point_B (numpy.ndarray): [x, y, z]

    Returns:
        float: Euclidean distance between two points.

    Raises:
        ValueError: If any of the two points does not have 3 values.
    """
    if len(point_A) != 3:
        raise ValueError("point needs to have 3 values")
    if len(point_B) != 3:
        raise ValueError("point needs to have 3 values")
    return np.sqrt(
        (point_B[0] - point_A[0]) ** 2
        + (point_B[1] - point_A[1]) ** 2
        + (point_B[2] - point_A[2]) ** 2
    )
