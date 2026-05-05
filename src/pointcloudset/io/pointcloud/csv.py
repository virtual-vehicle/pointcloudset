from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pointcloudset.io.pointcloud.delimited import read_delimited_coordinates, write_delimited_coordinates

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def read_csv(file_path: Path | str, **kwargs):
    """Read a CSV pointcloud file into a dataframe.

    Args:
        file_path: Path to the CSV file.
        **kwargs: Additional keyword arguments forwarded to the shared
            delimited reader. Supports ``normalize_xyz`` to opt in to
            converting ``X/Y/Z`` headers to ``x/y/z``.

    Returns:
        A dataframe containing pointcloud columns.

    Raises:
        ValueError: If parsing fails or required coordinate columns are missing.
    """
    return read_delimited_coordinates(
        file_path,
        format_name="CSV",
        default_sep=None,
        fallback_sep=None,
        normalize_xyz=kwargs.pop("normalize_xyz", False),
        **kwargs,
    )


def write_csv(pointcloud: PointCloud, file_path: Path, header: bool = True, sep: str = ",") -> None:
    """Write a pointcloud to a CSV file.

    Args:
        pointcloud: PointCloud instance to serialize.
        file_path: Destination file path.
        header: Whether to write a header row. Default is True for CSV files. Same as pandas.
        sep: Field delimiter.
    """
    write_delimited_coordinates(pointcloud, file_path, header=header, sep=sep)
