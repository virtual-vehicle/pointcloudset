"""
Routines for calculating differences and distances between entities.
"""

from .origin import calculate_distance_to_origin
from .plane import calculate_distance_to_plane
from .frame import calculate_all_point_differences
from .point import calculate_single_point_difference

ALL_DIFF = {
    "frame": calculate_all_point_differences,
    "plane": calculate_distance_to_plane,
    "point": calculate_single_point_difference,
    "origin": calculate_distance_to_origin,
}