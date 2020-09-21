import math

import numpy as np


def distance_to_point(point: np.array, plane_model: np.array) -> float:
    """Calculate the distance of a point to a plane
    https://mathworld.wolfram.com/Point-PlaneDistance.html

    Args:
        point (np.array): [x, y, z]
        plane (np.array): [a, b, c, d], could be provided by plane_segmentation

    Returns:
        [float]: distance to the point
    """
    if len(point) != 3:
        raise ValueError("point needs to have 3 values")
    if len(plane_model) != 4:
        raise ValueError("plane_model needs to have 4 values")
    return (
        plane_model[0] * point[0]
        + plane_model[1] * point[1]
        + plane_model[2] * point[2]
        + plane_model[3]
    ) / (math.sqrt(plane_model[0] ** 2 + plane_model[1] ** 2 + plane_model[2] ** 2))
