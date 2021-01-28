import numpy as np
import pandas as pd


def calculate_distance_to_point(frame, target: np.ndarray, **kwargs):
    """For each point in the pointcloud calculate the euclidian distance
    to the point. Adds a new column to the data with the values.
    """
    point_a = target
    points = frame.points.xyz
    distances = np.array([np.linalg.norm(point_a - point) for point in points])
    point_str = np.array2string(target, formatter={"float_kind": lambda x: "%.4f" % x})
    frame._add_column(f"distance to point: {point_str}", distances)
    return frame
