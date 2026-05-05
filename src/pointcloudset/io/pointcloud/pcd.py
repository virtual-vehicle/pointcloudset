from __future__ import annotations

import re
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


def _sanitize_pcd_field_names(field_names: list[str]) -> list[str]:
    sanitized: list[str] = []
    changed = False
    for name in field_names:
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
        if not safe or safe[0].isdigit():
            safe = f"f_{safe}"
        if safe != str(name):
            changed = True
        sanitized.append(safe)

    if len(set(sanitized)) != len(sanitized):
        raise ValueError(
            "PCD writing produced duplicate field names after sanitization. "
            "Please rename columns so they map to unique [A-Za-z_][A-Za-z0-9_]* names."
        )

    if changed:
        import warnings

        warnings.warn(
            "PCD field names were sanitized to be tool-compatible: only [A-Za-z_][A-Za-z0-9_]* are written.",
            UserWarning,
            stacklevel=3,
        )

    return sanitized


def _to_pcd_numeric_array(series: pd.Series, field_name: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise TypeError(f"PCD writing does not support non-numeric column '{field_name}'") from exc

    # Nullable pandas dtypes (e.g. Int64/boolean) need explicit ndarray conversion.
    if pd.api.types.is_extension_array_dtype(numeric.dtype):
        values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        values = numeric.to_numpy()

    values = np.asarray(values)
    if values.dtype.kind in {"O", "S", "U"}:
        raise TypeError(f"PCD writing does not support non-numeric column '{field_name}'")
    return values


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
    pcd_fields = _sanitize_pcd_field_names([str(field) for field in fields])
    arrays = []
    types = []
    for field in fields:
        values = _to_pcd_numeric_array(df[field], str(field))
        arrays.append(values)
        types.append(np.dtype(values.dtype))

    pcd = PcdPointCloud.from_points(arrays, fields=pcd_fields, types=types)
    pcd.save(Path(file_path))
