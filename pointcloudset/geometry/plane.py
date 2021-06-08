"""
Utility functions for geometric calculations with planes.
"""

import math

import numpy as np

from pointcloudset.geometry import point


def distance_to_point(
    point_A: np.ndarray, plane_model: np.ndarray, normal_dist: bool = True
) -> float:
    """Calculate the distance from a plane to a point.
    https://mathworld.wolfram.com/Point-PlaneDistance.html

    Note:
        Uses the plane equation a x + b y + c z + d = 0.

    Args:
        point_A (numpy.ndarray): [x, y, z], point for which the distance is calculated
            to plane.
        plane_model (numpy.ndarray):  [a, b, c, d] parameters of the plane equation,
            could be provided by :func:`pointcloudset.pointcloud.PointCloud.plane_segmentation`.
        normal_dist (bool): Calculate normal distance if ``True``, calculate
            distance in direction of line of sight if ``False``. Defaults to ``True``.

    Returns:
        float: Distance between plane and point.

    Raises:
        ValueError: If point does not have 3 values or if plane does not have 4 values.
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


def intersect_line_of_sight(line: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    """Calculate the point of intersection between a line and a plane.

    Note:
        Uses the plane equation a x + b y + c z + d = 0.

    Args:
        line (numpy.ndarray): [lx, ly, lz], line of sight through origin and point
            (lx,ly,lz).
        plane_model (numpy.ndarray): [a, b, c, d] parameters of the plane equation,
            could be provided by :func:`pointcloudset.pointcloud.PointCloud.plane_segmentation`.

    Returns:
        numpy.ndarray: [px, py, pz], point of intersection of line of sight and plane.

    Raises:
        ValueError: If line does not have 3 values or if plane does not have 4 values.
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
