import open3d as o3d
import pandas as pd
import pyntcloud

from typing import TYPE_CHECKING


def convert_df2pcd(df: pd.DataFrame) -> o3d.open3d_pybind.geometry.PointCloud:
    """Converts pandas dataframe to open3d PointCloud.

    Args:
        df (pd.DataFrame): pointcoud dataframe with x,y,z,intensity

    Returns:
        o3d.open3d_pybind.geometry.PointCloud: open3d pointcloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].to_numpy())
    return pcd


def convert_df2frame(df: pd.DataFrame):
    pass


def convert_open3d2frame(df: pd.DataFrame):
    pass


def convert_pyntcloud2frame(pyntcloud_in: pyntcloud.PyntCloud):
    """Converts a pyntcloud to a lidar Frame.

    Args:
        pyntcloud_in (pyntcloud.PyntCloud): pyntcloud object to convert to frame

    Returns:
        Frame: Frame object from pyntcloud
    """
    data = pyntcloud_in.points
    return Frame(data=data)