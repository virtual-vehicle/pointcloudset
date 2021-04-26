"""
Functions to calculate distances of points in frame to origin.
"""

import numpy as np

from lidar.diff.point import calculate_distance_to_point


def calculate_distance_to_origin(frame, **kwargs):
    """Calculate the Euclidian distance to the origin (0,0,0) for each point in the frame.

    Note:
        Adds the result as a new column to the data of the frame.

    Args:
        frame (Frame): Frame for which the Euclidean distance to the origin is
            calculated.

    Returns:
        Frame: Frame including Euclidean distances to the origin for each point.
    """
    point_a = np.array((0, 0, 0))

    return calculate_distance_to_point(frame, target=point_a)
