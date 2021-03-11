"""
Routines for calculating differences and distances between entities.
"""

from .frame import calculate_distance_to_frame
from .origin import calculate_distance_to_origin
from .plane import calculate_distance_to_plane
from .point import calculate_distance_to_point

ALL_DIFFS = {
    "frame": calculate_distance_to_frame,
    "plane": calculate_distance_to_plane,
    "point": calculate_distance_to_point,
    "origin": calculate_distance_to_origin,
}
