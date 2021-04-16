"""
Functions to calculate distances of points in frame to origin.
"""

import numpy as np

from lidar.diff.point import calculate_distance_to_point


def calculate_distance_to_origin(frame, **kwargs):
    """For each point in the frame calculate the euclidian distance
    to the origin (0,0,0). Adds a new column to the data with the values.

    Args:
        frame (Frame): Frame for which the distance to the origin is calculated.

    Returns:
        Frame: Frame with euclidean distance to origin for each point.
    """
    point_a = np.array((0, 0, 0))

    return calculate_distance_to_point(frame, target=point_a)
