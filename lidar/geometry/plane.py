"""
Utility functions for geometric calculations with planes.
"""

import math

import numpy as np

from lidar.geometry import point


def distance_to_point(
    point_A: np.array, plane_model: np.array, normal_dist: bool = True
) -> float:
    """Calculate the distance of a point to a plane. Uses the plane equation a x + b y + c z + d = 0
    https://mathworld.wolfram.com/Point-PlaneDistance.html

    Args:
        point_A (np.array): [x, y, z]
        plane_model (np.array): [a, b, c, d], could be provided by plane_segmentation
        normal_dist (bool): Calculate normal distance if True, calculate
            distance in direction of line of sight if False. Defaults to True.

    Returns:
        [float]: Distance to the point.
    """
    if len(point_A) != 3:
        raise ValueError("point needs to have 3 values")
    if len(plane_model) != 4:
        raise ValueError("plane_model needs to have 4 values")
    if normal_dist:
        distance = (
            plane_model[0] * point_A[0]
            + plane_model[1] * point_A[1]
            + plane_model[2] * point_A[2]
            + plane_model[3]
        ) / (math.sqrt(plane_model[0] ** 2 + plane_model[1] ** 2 + plane_model[2] ** 2))
    else:
        line_of_sight = point_A
        intersection_point = intersect_line_of_sight(line_of_sight, plane_model)
        distance = point.distance_to_point(intersection_point, point_A)
    return distance


def intersect_line_of_sight(line: np.array, plane_model: np.array) -> np.array:
    """Calculate the point of intersection between a line and a plane. Uses the plane equation a x + b y + c z + d = 0.

    Args:
        line (np.array): [lx, ly, lz]
        plane (np.array): [a, b, c, d], could be provided by plane_segmentation

    Returns:
        [np.array]: Point of intersection.
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
