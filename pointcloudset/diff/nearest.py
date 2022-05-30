"""
Functions to calculate differences between the pointcloud nearest points in another one.
"""
import numpy as np


def calculate_distance_to_nearest(pointcloud, target):
    pc1 = pointcloud.to_instance("open3d")
    pc2 = target.to_instance("open3d")
    distance = pc1.compute_point_cloud_distance(pc2)
    pointcloud._add_column("distance to nearest point", np.array(distance))
    return pointcloud
