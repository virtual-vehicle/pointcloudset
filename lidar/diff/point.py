"""
Functions to calculate distances between points in frame and a point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lidar


def calculate_distance_to_point(
    frame: lidar.Frame, target: np.ndarray, **kwargs
) -> lidar.Frame:
    """For each point in the frame calculate the euclidian distance
    to the point. Adds a new column to the data with the values.

    Args:
        target (np.ndarray): Point to which the distance calculated.

    Returns:
        Frame: Frame with distances to a point for each point.
    """
    point_a = target
    points = frame.points.xyz
    distances = np.array([np.linalg.norm(point_a - point) for point in points])
    point_str = np.array2string(target, formatter={"float_kind": lambda x: "%.4f" % x})
    frame._add_column(f"distance to point: {point_str}", distances)
    return frame
