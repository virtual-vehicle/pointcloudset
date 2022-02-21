"""
Utility function for filtering frames based on statistics.
"""


from typing import TYPE_CHECKING

from pointcloudset.config import OPS

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def quantile_filter(
    pointcloud: PointCloud,
    dim: str,
    relation: str = ">=",
    cut_quantile: float = 0.5,
) -> PointCloud:
    """Filtering based on quantile values of dimension dim of the data.

    Args:
        pointcloud (PointCloud): PointCloud to be filtered.
        dim (str): Dimension to limit. Any column in data (not just x, y or z, but also
            "intensity")
        relation (str, optional): Any operator as string. Defaults to ">=".
        cut_quantile (float, optional): Quantile to compare to. Defaults to 0.5.

    Returns:
        PointCloud: PointCloud which fullfils the criteria.
    """
    cut_value = pointcloud.data[dim].quantile(cut_quantile)
    filter_array = OPS[relation](pointcloud.data[dim], cut_value)
    return pointcloud.apply_filter(filter_array.to_numpy())


def value_filter(
    pointcloud: PointCloud,
    dim: str,
    relation: str,
    value: float,
) -> PointCloud:
    """Limit the range of certain values in a PointCloud.

    Args:
        pointcloud (PointCloud): PointCloud to be filtered.
        dim (str): Dimension to limit. Any column in data (not just x, y or z, but also
            "intensity")
        relation (str): Any operator as string.
        value (float): Value to limit.

    Returns:
        PointCloud: PointCloud which fullfils the criteria.
    """

    bool_array = (OPS[relation](pointcloud.data[dim], value)).to_numpy()
    return pointcloud.apply_filter(bool_array)


def remove_radius_outlier(
    pointcloud: PointCloud, nb_points: int, radius: float
) -> PointCloud:
    """Function to remove points that have less than nb_points in a given
    sphere of a given radius.

    Args:
        pointcloud (PointCloud): PointCloud from which to remove points.
        nb_points (int): Number of points within the radius.
        radius (float): Radius of the sphere.

    Returns:
        PointCloud: PointCloud without outliers.
    """
    pcd = pointcloud.to_instance("open3d")
    _, index_to_keep = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return pointcloud.apply_filter(index_to_keep)
