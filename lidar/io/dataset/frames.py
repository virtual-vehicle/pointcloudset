from __future__ import annotations

from typing import TYPE_CHECKING, List

import dask

if TYPE_CHECKING:
    from lidar import Frame


def dataset_from_frames(frames: List[Frame]) -> dict:
    """Convert a list of lidar frames to a new dataset.

    Args:
        frames (List[Frame]): A list of frames.

    Returns:
        dict: For convertion to dataset.
    """
    data = [dask.delayed(frame.data) for frame in frames]
    timestamps = [frame.timestamp for frame in frames]
    meta = {"orig_file": "from frames list"}
    return {"data": data, "timestamps": timestamps, "meta": meta}
