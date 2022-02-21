from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def from_dataframe(df: pd.DataFrame) -> dict:
    """Converts pandas DataFrame to a PointCloud.

    Args:
        df (pandas.DataFrame): Pandas DataFrame to convert.

    Returns:
        dict: Returns data for PointCloud.
    """
    return {"data": df}


def to_dataframe(pointcloud: PointCloud) -> pd.DataFrame:
    """Converts a PointCloud to a pandas DataFrame.

    Args:
        pointcloud (PointCloud): PointCloud to convert.

    Returns:
        pandas.DataFrame: Data of PointCloud.
    """
    return pointcloud.data
