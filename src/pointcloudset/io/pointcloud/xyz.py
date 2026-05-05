from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pointcloudset.io.pointcloud.delimited import read_delimited_coordinates, write_delimited_coordinates

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def read_xyz(file_path: Path | str, **kwargs) -> pd.DataFrame:
    return read_delimited_coordinates(
        file_path,
        format_name="XYZ",
        default_sep=None,
        fallback_sep=r"\s+",
        normalize_xyz=kwargs.pop("normalize_xyz", False),
        **kwargs,
    )


def write_xyz(pointcloud: PointCloud, file_path: Path, header: bool = False, sep: str = " ") -> None:
    write_delimited_coordinates(pointcloud, file_path, header=header, sep=sep, columns=["x", "y", "z"])
