from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pointcloudset.io.pointcloud.delimited import read_delimited_coordinates, write_delimited_coordinates

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def read_xyz(file_path: Path | str, **kwargs) -> pd.DataFrame:
    """Read an XYZ pointcloud file into a dataframe.

    Args:
        file_path: Path to the XYZ file.
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
        format_name="XYZ",
        default_sep=None,
        fallback_sep=r"\s+",
        normalize_xyz=kwargs.pop("normalize_xyz", False),
        **kwargs,
    )


def write_xyz(pointcloud: PointCloud, file_path: Path, header: bool = False, sep: str = " ") -> None:
    """Write a pointcloud to an XYZ file.

    Args:
        pointcloud: PointCloud instance to serialize.
        file_path: Destination file path.
        header: Whether to write a header row. Default is False for XYZ files.
        sep: Field delimiter.
    """
    write_delimited_coordinates(pointcloud, file_path, header=header, sep=sep, columns=["x", "y", "z"])
