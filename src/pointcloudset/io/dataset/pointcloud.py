from __future__ import annotations

from typing import TYPE_CHECKING

import dask

if TYPE_CHECKING:
    from src import PointCloud


def dataset_from_pointclouds(pointclouds: list[PointCloud]) -> dict:
    """Convert a list of pointcloud pointclouds to a new dataset.

    Args:
        pointclouds (list[PointCloud]): A list of pointclouds.

    Returns:
        dict: For convertion to dataset.
    """
    data = [dask.delayed(pointcloud.data) for pointcloud in pointclouds]
    timestamps = [pointcloud.timestamp for pointcloud in pointclouds]
    meta = {"orig_file": "from pointclouds list"}
    return {"data": data, "timestamps": timestamps, "meta": meta}
