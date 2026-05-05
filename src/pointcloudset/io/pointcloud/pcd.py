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
    """Drop internal padding placeholder columns from parsed PCD data.

    Args:
        df: Dataframe created from ``pypcd4`` point records.

    Returns:
        Dataframe without placeholder padding columns.
    """
    # pypcd4 preserves padding bytes as generated placeholder field names; they are not useful user data.
    keep_columns = [col for col in df.columns if not str(col).startswith("#$%&~~")]
    return df.loc[:, keep_columns]


def _sanitize_pcd_field_names(field_names: list[str]) -> list[str]:
    """Convert dataframe column names to tool-compatible PCD field names.

    Args:
        field_names: Original dataframe column names.

    Returns:
        Sanitized field names matching ``[A-Za-z_][A-Za-z0-9_]*``.

    Raises:
        ValueError: If sanitization causes duplicate output names.
    """
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
    """Convert a dataframe column to a numeric ndarray for PCD writing.

    Args:
        series: Column values to convert.
        field_name: Column name used in error messages.

    Returns:
        Numeric NumPy array suitable for PCD serialization.

    Raises:
        TypeError: If the column cannot be represented as numeric data.
    """
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


def _normalize_xyz_pcd_columns(df: pd.DataFrame, normalize_xyz: bool, file_path: Path) -> pd.DataFrame:
    """Validate and optionally normalize PCD coordinate column casing.

    Args:
        df: Parsed PCD dataframe.
        normalize_xyz: Whether to convert ``X/Y/Z`` to ``x/y/z``.
        file_path: Source file path used in error messages.

    Returns:
        Dataframe with lowercase coordinate columns.

    Raises:
        ValueError: If coordinate columns are missing or uppercase coordinates
            are provided while normalization is disabled.
    """
    if {"x", "y", "z"}.issubset(df.columns):
        return df

    mapping: dict[str, str] = {}
    for col in df.columns:
        col_str = str(col)
        lowered = col_str.lower()
        if lowered in {"x", "y", "z"} and lowered not in mapping:
            mapping[lowered] = col_str

    if not {"x", "y", "z"}.issubset(mapping.keys()):
        raise ValueError("PCD file must contain x, y and z columns")

    if not normalize_xyz:
        raise ValueError(
            f"PCD file '{file_path}' contains coordinate columns X/Y/Z. "
            "pointcloudset expects lowercase x/y/z internally. "
            "Pass normalize_xyz=True to convert X/Y/Z to x/y/z."
        )

    rename_map = {original: lowered for lowered, original in mapping.items() if original != lowered}
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def read_pcd(file_path: Path | str, normalize_xyz: bool = False) -> pd.DataFrame:
    """Read a PCD pointcloud file into a dataframe.

    Args:
        file_path: Path to the PCD file.
        normalize_xyz: Whether to convert ``X/Y/Z`` headers to ``x/y/z``.

    Returns:
        A dataframe containing pointcloud columns.

    Raises:
        ValueError: If required coordinate columns are missing or normalization
            is required but not enabled.
    """
    path = Path(file_path)
    pointcloud = PcdPointCloud.from_path(path)
    df = pd.DataFrame.from_records(pointcloud.pc_data)
    df = _sanitize_pcd_columns(df)
    return _normalize_xyz_pcd_columns(df, normalize_xyz=normalize_xyz, file_path=path)


def write_pcd(pointcloud: PointCloud, file_path: Path) -> None:
    """Write a pointcloud to a PCD file.

    Args:
        pointcloud: PointCloud instance to serialize.
        file_path: Destination file path.

    Raises:
        ValueError: If the pointcloud is missing ``x``, ``y``, or ``z`` columns.
        TypeError: If one or more columns cannot be converted to numeric arrays.
    """
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
