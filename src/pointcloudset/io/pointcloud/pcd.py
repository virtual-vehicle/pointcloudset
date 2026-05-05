from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pypcd4 import PointCloud as PcdPointCloud

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def _sanitize_pcd_columns(df: pd.DataFrame) -> pd.DataFrame:
    # pypcd4 preserves padding bytes as generated placeholder field names; they are not useful user data.
    keep_columns = [col for col in df.columns if not str(col).startswith("#$%&~~")]
    return df.loc[:, keep_columns]


def read_pcd(file_path: Path | str) -> pd.DataFrame:
    pointcloud = PcdPointCloud.from_path(Path(file_path))
    df = pd.DataFrame.from_records(pointcloud.pc_data)
    df = _sanitize_pcd_columns(df)
    if not {"x", "y", "z"}.issubset(df.columns):
        raise ValueError("PCD file must contain x, y and z columns")
    return df


def write_pcd(pointcloud: PointCloud, file_path: Path) -> None:
    df = pointcloud.data
    if not {"x", "y", "z"}.issubset(df.columns):
        raise ValueError("PointCloud must have x, y and z columns")

    fields = list(df.columns)
    arrays = []
    types = []
    for field in fields:
        values = df[field].to_numpy()
        if values.dtype.kind in {"O", "S", "U"}:
            raise TypeError(f"PCD writing does not support non-numeric column '{field}'")
        arrays.append(values)
        types.append(np.dtype(values.dtype))

    pcd = PcdPointCloud.from_points(arrays, fields=fields, types=types)
    pcd.save(Path(file_path))
