from __future__ import annotations

from typing import TYPE_CHECKING, List

import dask

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def dataset_from_frames(frames: List[PointCloud]) -> dict:
    """Convert a list of pointcloud frames to a new dataset.

    Args:
        frames (List[PointCloud]): A list of frames.

    Returns:
        dict: For convertion to dataset.
    """
    data = [dask.delayed(pointcloud.data) for pointcloud in frames]
    timestamps = [pointcloud.timestamp for pointcloud in frames]
    meta = {"orig_file": "from frames list"}
    return {"data": data, "timestamps": timestamps, "meta": meta}
