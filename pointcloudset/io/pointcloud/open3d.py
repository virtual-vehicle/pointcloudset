from __future__ import annotations

from typing import TYPE_CHECKING

import open3d as o3d
from pyntcloud import PyntCloud

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def from_open3d(open3d_data: o3d.open3d_pybind.geometry.PointCloud) -> dict:
    """Converts a open3d pointcloud to a PointCloud.

    Args:
        open3d_data (open3d.open3d_pybind.geometry.PointCloud): Open3D pointcloud which should be converted.

    Returns:
        dict: Pointcloud data.
    """
    pyntcloud_data = PyntCloud.from_instance("open3d", open3d_data)
    return {"data": pyntcloud_data.points}


def to_open3d(pointcloud: PointCloud) -> o3d.open3d_pybind.geometry.PointCloud:
    """Converts PointCloud to open3d PointCloud.

    Args:
        df (pd.DataFrame): Pointcloud dataframe with x,y,z,intensity.

    Returns:
        open3d.open3d_pybind.geometry.PointCloud: Open3d pointcloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.points.xyz)
    return pcd
