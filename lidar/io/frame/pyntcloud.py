from __future__ import annotations

from typing import TYPE_CHECKING

import pyntcloud

if TYPE_CHECKING:
    from lidar import Frame


def from_pyntcloud(pyntcloud_data: pyntcloud.PyntCloud) -> dict:
    """Converts a pyntcloud pointcloud to a Frame.

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


def to_pyntcloud(frame: Frame) -> pyntcloud.PyntCloud:
    """Converts a Frame to a pyntcloud pointcloud.

    Args:
        frame (Frame): Frame which should be converted.

    Returns:
        pyntcloud.PointCloud: Pyntcloud pointcloud.
    """
    return frame.points
