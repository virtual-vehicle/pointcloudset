"""
# Difference to coordinate system orgin.
"""

import numpy as np


def calculate_distance_to_origin(frame):
    """For each point in the pointcloud calculate the euclidian distance
    to the origin (0,0,0). Adds a new column to the data with the values.
    """
    point_a = np.array((0.0, 0.0, 0.0))
    points = frame.points.xyz
    distances = np.array([np.linalg.norm(point_a - point) for point in points])
    frame._add_column("distance to origin", distances)
    return frame
