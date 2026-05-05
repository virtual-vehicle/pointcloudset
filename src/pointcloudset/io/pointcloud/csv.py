from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pointcloudset.io.pointcloud.delimited import read_delimited_coordinates, write_delimited_coordinates

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def read_csv(file_path: Path | str, **kwargs):
    return read_delimited_coordinates(
        file_path,
        format_name="CSV",
        default_sep=None,
        fallback_sep=None,
        normalize_xyz=kwargs.pop("normalize_xyz", False),
        **kwargs,
    )


def write_csv(pointcloud: PointCloud, file_path: Path, header: bool = True, sep: str = ",") -> None:
    """Exports the pointcloud as a csv for use with cloud compare or similar tools.
    Currently not all attributes of a pointcloud are saved so some information is lost when
    using this function.

    Args:
        file_path (Path): Destination.
    """
    write_delimited_coordinates(pointcloud, file_path, header=header, sep=sep)
