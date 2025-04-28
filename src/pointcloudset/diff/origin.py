"""
Functions to calculate distances of points in pointcloud to origin.
"""

import numpy as np

from pointcloudset.diff.point import calculate_distance_to_point


def calculate_distance_to_origin(pointcloud, **kwargs):
    """Calculate the Euclidian distance to the origin (0,0,0) for each point in the pointcloud.

    Note:
        Adds the result as a new column to the data of the pointcloud.

    Args:
        pointcloud (PointCloud): PointCloud for which the Euclidean distance to the origin is
            calculated.

    Returns:
        PointCloud: PointCloud including Euclidean distances to the origin for each point.
    """
    point_a = np.array((0, 0, 0))

    return calculate_distance_to_point(pointcloud, target=point_a)
