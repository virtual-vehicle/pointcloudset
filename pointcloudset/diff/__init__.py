"""
Functions to calculate differences and distances between entities.
"""

from pointcloudset.diff.frame import calculate_distance_to_frame
from pointcloudset.diff.origin import calculate_distance_to_origin
from pointcloudset.diff.plane import calculate_distance_to_plane
from pointcloudset.diff.point import calculate_distance_to_point

ALL_DIFFS = {
    "frame": calculate_distance_to_frame,
    "plane": calculate_distance_to_plane,
    "point": calculate_distance_to_point,
    "origin": calculate_distance_to_origin,
}
