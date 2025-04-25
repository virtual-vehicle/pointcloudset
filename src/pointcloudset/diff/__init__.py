"""
Functions to calculate differences and distances between entities.
"""

from src.diff.nearest import calculate_distance_to_nearest
from src.diff.origin import calculate_distance_to_origin
from src.diff.plane import calculate_distance_to_plane
from src.diff.point import calculate_distance_to_point
from src.diff.pointcloud import calculate_distance_to_pointcloud

ALL_DIFFS = {
    "pointcloud": calculate_distance_to_pointcloud,
    "plane": calculate_distance_to_plane,
    "point": calculate_distance_to_point,
    "origin": calculate_distance_to_origin,
    "nearest": calculate_distance_to_nearest,
}
