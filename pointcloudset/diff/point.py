"""
Functions to calculate distances between points in pointcloud and a point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pointcloudset


def calculate_distance_to_point(
    pointcloud: pointcloudset.pointcloud.PointCloud, target: np.ndarray, **kwargs
) -> pointcloudset.pointcloud.PointCloud:
    """Calculate the Euclidian distance to a point for each point in the pointcloud.

    Note:
        Adds the result as a new column to the data of the pointcloud.

    Args:
        pointcloud (PointCloud): PointCloud for which the Euclidean distance to the point is
            calculated.
        target (numpy.ndarray): [x, y, z] as coordinates of the point to which the
            Euclidean distance is calculated.

    Returns:
        PointCloud: PointCloud including Euclidean distances to a point for each point.
    """
    point_a = target
    points = pointcloud.points.xyz
    distances = np.array([np.linalg.norm(point_a - point) for point in points])
    point_str = np.array2string(target, formatter={"float_kind": lambda x: "%.4f" % x})
    point_str = " ".join(point_str.split())  # delete multiple white space
    pointcloud._add_column(f"distance to point: {point_str}", distances)
    return pointcloud
