import numpy as np
import open3d as o3d
import pandas as pd


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
