import math

import numpy as np


def distance_to_point(point: np.array, plane_model: np.array) -> float:
    """Calculate the distance of a point to a plane. Uses the plane equation a x + b y + c z + d = 0
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


def intersect_line_of_sight(line: np.array, plane_model: np.array) -> np.array:
    """Calculate the point of intersection between a line and a plane. Uses the plane equation a x + b y + c z + d = 0

    Args:
        line (np.array): [lx, ly, lz]
        plane (np.array): [a, b, c, d], could be provided by plane_segmentation

    Returns:
        [np.array]: point of intersection
    """
    if len(line) != 3:
        raise ValueError("line needs to have 3 values")
    if len(plane_model) != 4:
        raise ValueError("plane_model needs to have 4 values")
    t = (-plane_model[3]) / (
        line[0] * plane_model[0] + line[1] * plane_model[1] + line[2] * plane_model[2]
    )
    px = line[0] * t
    py = line[1] * t
    pz = line[2] * t

    return np.array([px, py, pz])
