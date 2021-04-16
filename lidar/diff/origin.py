"""
Difference to coordinate system origin.

"""

import numpy as np

from lidar.diff.point import calculate_distance_to_point


def calculate_distance_to_origin(frame, **kwargs):
    """For each point in the frame calculate the euclidian distance
    to the origin (0,0,0). Adds a new column to the data with the values.

    Args:
        frame (Frame): Frame for which the distance to the origin is calculated.
    """
    point_a = np.array((0, 0, 0))

    return calculate_distance_to_point(frame, target=point_a)
