"""
Functions to calculate distances of points in frame and a plane.
"""

import numpy as np

from lidar.geometry import plane


def calculate_distance_to_plane(
    frame, target: np.array, absolute_values: bool = True, normal_dist: bool = True
):
    """Calculate the distance of each point to a plane.

    Note:
        Adds the result as a new column to the data of the frame. Uses the plane
        equation a x + b y + c z + d = 0.

    Args:
        frame (Frame): Frame for which the distance to the plane is calculated.
        target (numpy.ndarray): [a, b, c, d] as parameters for the target plane, could be
            provided by plane_segmentation.
        absolute_values (bool, optional): Calculate absolute distances if ``True``.
            Defaults to ``True``.
        normal_dist (bool, optional): Calculate normal distance if ``True``, calculate
            distance in direction of line of sight if ``False``. Defaults to ``True``.

    Returns:
        Frame: Frame with distance to a plane for each point.
    """
    points = frame.points.xyz
    distances = np.asarray(
        [plane.distance_to_point(point, target, normal_dist) for point in points]
    )
    if absolute_values:
        distances = np.absolute(distances)
    plane_str = np.array2string(target, formatter={"float_kind": lambda x: "%.4f" % x})
    plane_str = " ".join(plane_str.split())  # delete multiple white space
    frame._add_column(f"distance to plane: {plane_str}", distances)
    return frame
