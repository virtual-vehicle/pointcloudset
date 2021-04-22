"""
Utility function for filtering frames based on statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lidar.config import OPS

if TYPE_CHECKING:
    from lidar import Frame


def quantile_filter(
    frame, dim: str, relation: str = ">=", cut_quantile: float = 0.5
) -> lidar.Frame:
    """Filtering based on quantile values of dimension dim of the data.

    Example:
        .. code-block:: python

            testframe.filter("quantile","intensity","==",0.5)

    Args:
        frame (Frame): Frame to be filtered
        dim (str): Dimension to limit, column in data of frame, for example "intensity"
        relation (str, optional): Any operator as string. Defaults to ">=".
        cut_quantile (float, optional): Quantile to compare to. Defaults to 0.5.

    Returns:
        Frame: Frame which fullfils the criteria.
    """
    cut_value = frame.data[dim].quantile(cut_quantile)
    filter_array = OPS[relation](frame.data[dim], cut_value)
    return frame.apply_filter(filter_array.to_numpy())


def value_filter(frame, dim: str, relation: str, value: float) -> lidar.Frame:
    """Limit the range of certain values in lidar Frame.

    Example:
        .. code-block:: python

            testframe.filter("value", "x", ">", 1.0)

    Args:
        frame (Frame): Frame to be filtered.
        dim (str): Dimension to limit, any column in data not just x, y, or z, for example "intensity"
        relation (str): Any operator as string.
        value (float): Value to limit.

    Returns:
        Frame: Frame which fullfils the criteria.
    """

    bool_array = (OPS[relation](frame.data[dim], value)).to_numpy()
    return frame.apply_filter(bool_array)


def remove_radius_outlier(frame: Frame, nb_points: int, radius: float) -> lidar.Frame:
    """Function to remove points that have less than nb_points in a given
    sphere of a given radius.

    Args:
        frame (Frame): Frame from which to remove points.
        nb_points (int): Number of points within the radius.
        radius (float): Radius of the sphere.

    Returns:
        Frame: Frame without outliers.
    """
    pcd = frame.to_instance("open3d")
    _, index_to_keep = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return frame.apply_filter(index_to_keep)
