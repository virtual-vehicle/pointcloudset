from __future__ import annotations

from typing import TYPE_CHECKING

import pyntcloud

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def from_pyntcloud(pyntcloud_data: pyntcloud.PyntCloud) -> dict:
    """Converts a pyntcloud pointcloud to a PointCloud.

    Args:
        pyntcloud_data (pyntcloud.PyntCloud): Pyntcloud pointcloud which should be converted.

    Returns:
        dict: Pointcloud data.
    """
    if not isinstance(pyntcloud_data, pyntcloud.PyntCloud):
        raise TypeError(
            f"Type {type(pyntcloud_data)} not supported for conversion."
            f"Expected pyntcloud.PyntCloud"
        )
    return {"data": pyntcloud_data.points}


def to_pyntcloud(pointcloud: PointCloud) -> pyntcloud.PyntCloud:
    """Converts a PointCloud to a pyntcloud pointcloud.

    Args:
        pointcloud (PointCloud): PointCloud which should be converted.

    Returns:
        pyntcloud.PointCloud: Pyntcloud pointcloud.
    """
    return pointcloud.points
