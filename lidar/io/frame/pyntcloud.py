from __future__ import annotations

from typing import TYPE_CHECKING

import pyntcloud

if TYPE_CHECKING:
    from lidar import Frame


def from_pyntcloud(pyntcloud_data: pyntcloud.PyntCloud) -> dict:
    if not isinstance(pyntcloud_data, pyntcloud.PyntCloud):
        raise TypeError(
            f"Type {type(pyntcloud_data)} not supported for conversion."
            f"Expected pyntcloud.PyntCloud"
        )
    return {"data": pyntcloud_data.points}


def to_pyntcloud(frame: Frame) -> pyntcloud.PyntCloud:
    return frame.points
