"""
Utility functions for geometric calculations with points.
"""

import numpy as np


def distance_to_point(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Calculate the Euclidean distance of a point to another point.

    Args:
        point_a (numpy.ndarray): [x, y, z]
        point_b (numpy.ndarray): [x, y, z]

    Returns:
        float: Euclidean distance between two points.

    Raises:
        ValueError: If any of the two points does not have 3 values.
    """
    if len(point_a) != 3:
        raise ValueError("point needs to have 3 values")
    if len(point_b) != 3:
        raise ValueError("point needs to have 3 values")
    return np.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2 + (point_b[2] - point_a[2]) ** 2)
