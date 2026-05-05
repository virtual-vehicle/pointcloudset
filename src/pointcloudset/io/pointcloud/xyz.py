from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def _ensure_xyz_columns(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    if {"x", "y", "z"}.issubset(set(df.columns)):
        return df

    if df.shape[1] < 3:
        raise ValueError(f"XYZ file '{file_path}' must provide at least three columns for x, y, z")

    renamed_columns = ["x", "y", "z"] + [f"field_{i}" for i in range(3, df.shape[1])]
    df = df.copy()
    df.columns = renamed_columns
    warnings.warn(
        f"XYZ file '{file_path}' has no x/y/z header. Assuming first three columns are x, y, z.",
        UserWarning,
        stacklevel=2,
    )
    return df


def read_xyz(file_path: Path | str, **kwargs) -> pd.DataFrame:
    path = Path(file_path)

    if any(key in kwargs for key in ("sep", "delimiter", "header", "names")):
        return _ensure_xyz_columns(pd.read_csv(path, **kwargs), path)

    try:
        # Keep compatibility with CSV-style XYZ files when possible.
        df = pd.read_csv(path, **kwargs)
        if {"x", "y", "z"}.issubset(set(df.columns)):
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        if {"x", "y", "z"}.issubset(set(df.columns)):
            return df
    except Exception:
        pass

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return _ensure_xyz_columns(df, path)


def write_xyz(pointcloud: PointCloud, file_path: Path, header: bool = False, sep: str = " ") -> None:
    xyz = pointcloud.data[["x", "y", "z"]]
    xyz.to_csv(file_path, index=False, header=header, sep=sep)
