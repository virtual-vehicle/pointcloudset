"""
Functions to calculate differences between the pointcloud nearest points in another one.
"""

from scipy.spatial import KDTree


def calculate_distance_to_nearest(pointcloud, target):
    """Calculate the distance for each point in a pointcloud to the nearest points in
    the target pointcloud.

    Note:
        Adds the results to the data of the pointcloud.

    Args:
        pointcloud (PointCloud): PointCloud for which the differences to the target are
            calculated.
        target (PointCloud): PointCloud to calcluate the distances to.

    Returns:
        PointCloud: PointCloud with the colum "distance to nearest point".

    Raises:
        ValueError: If distance ot nearest points already exits.

    """
    if "distance to nearest point" in pointcloud.data.columns:
        raise ValueError("distance to nearest point already exists.")
    distances, _ = KDTree(target.points.xyz).query(pointcloud.points.xyz)
    pointcloud._add_column("distance to nearest point", distances)
    return pointcloud
