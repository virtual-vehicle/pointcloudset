"""
Routines for calculating differences and distances between entities.
"""

from .frame import calculate_all_point_differences
from .origin import calculate_distance_to_origin
from .plane import calculate_distance_to_plane
from .point import calculate_distance_to_point

ALL_DIFF = {
    "frame": calculate_all_point_differences,
    "plane": calculate_distance_to_plane,
    "point": calculate_distance_to_point,
    "origin": calculate_distance_to_origin,
}