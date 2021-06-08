from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def write_csv(pointcloud: PointCloud, file_path: Path) -> None:
    """Exports the pointcloud as a csv for use with cloud compare or similar tools.
    Currently not all attributes of a pointcloud are saved so some information is lost when
    using this function.

    Args:
        file_path (Path): Destination.
    """
    pointcloud.data.to_csv(file_path, index=False)