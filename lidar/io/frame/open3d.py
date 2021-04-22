from __future__ import annotations

from typing import TYPE_CHECKING

import open3d as o3d
from pyntcloud import PyntCloud

if TYPE_CHECKING:
    from lidar import Frame


def from_open3d(open3d_data: o3d.open3d_pybind.geometry.PointCloud) -> dict:
    pyntcloud_data = PyntCloud.from_instance("open3d", open3d_data)
    return {"data": pyntcloud_data.points}


def to_open3d(frame: Frame) -> o3d.open3d_pybind.geometry.PointCloud:
    """Converts pandas dataframe to open3d PointCloud.

    Args:
        df (pd.DataFrame): pointcloud dataframe with x,y,z,intensity

    Returns:
        o3d.open3d_pybind.geometry.PointCloud: open3d pointcloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame.points.xyz)
    return pcd
