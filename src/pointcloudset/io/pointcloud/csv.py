from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def read_csv(file_path: Path | str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(file_path, **kwargs)
    if not {"x", "y", "z"}.issubset(df.columns):
        raise ValueError("CSV file must contain x, y and z columns")
    return df


def write_csv(pointcloud: PointCloud, file_path: Path) -> None:
    """Exports the pointcloud as a csv for use with cloud compare or similar tools.
    Currently not all attributes of a pointcloud are saved so some information is lost when
    using this function.

    Args:
        file_path (Path): Destination.
    """
    pointcloud.data.to_csv(file_path, index=False)
