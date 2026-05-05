from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pointcloudset import PointCloud


def ensure_coordinate_columns(df: pd.DataFrame, file_path: Path, format_name: str) -> pd.DataFrame:
    if {"x", "y", "z"}.issubset(set(df.columns)):
        return df

    if df.shape[1] < 3:
        raise ValueError(f"{format_name} file '{file_path}' must provide at least three columns for x, y, z")

    renamed_columns = ["x", "y", "z"] + [f"field_{i}" for i in range(3, df.shape[1])]
    df = df.copy()
    df.columns = renamed_columns
    warnings.warn(
        f"{format_name} file '{file_path}' has no x/y/z header. Assuming first three columns are x, y, z.",
        UserWarning,
        stacklevel=2,
    )
    return df


def _raise_uppercase_xyz_error(file_path: Path, format_name: str) -> None:
    raise ValueError(
        f"{format_name} file '{file_path}' contains coordinate columns X/Y/Z. "
        "pointcloudset expects lowercase x/y/z internally. "
        "Pass normalize_xyz=True to convert X/Y/Z to x/y/z."
    )


def _normalize_xyz_columns(
    df: pd.DataFrame,
    *,
    normalize_xyz: bool,
    file_path: Path,
    format_name: str,
    allow_infer_coordinate_columns: bool = True,
) -> pd.DataFrame:
    if {"x", "y", "z"}.issubset(set(df.columns)):
        return df

    lower_to_original: dict[str, str] = {}
    for col in df.columns:
        col_str = str(col)
        lowered = col_str.lower()
        if lowered in {"x", "y", "z"} and lowered not in lower_to_original:
            lower_to_original[lowered] = col_str

    if {"x", "y", "z"}.issubset(lower_to_original.keys()):
        if not normalize_xyz:
            _raise_uppercase_xyz_error(file_path, format_name)

        rename_map = {original: lowered for lowered, original in lower_to_original.items() if original != lowered}
        if rename_map:
            return df.rename(columns=rename_map)
        return df

    if df.shape[1] < 3:
        raise ValueError(f"{format_name} file '{file_path}' must provide at least three columns for x, y, z")

    if not allow_infer_coordinate_columns:
        raise ValueError(
            f"{format_name} file '{file_path}' was read with explicit column names, but columns x, y, z are required."
        )

    return ensure_coordinate_columns(df, file_path, format_name)


def read_delimited_coordinates(
    file_path: Path | str,
    *,
    format_name: str,
    default_sep: str | None,
    fallback_sep: str | None = None,
    normalize_xyz: bool = False,
    **kwargs,
) -> pd.DataFrame:
    path = Path(file_path)

    # If users explicitly provide column names, keep those names untouched.
    if "names" in kwargs:
        df = pd.read_csv(path, **kwargs)
        return _normalize_xyz_columns(
            df,
            normalize_xyz=normalize_xyz,
            file_path=path,
            format_name=format_name,
            allow_infer_coordinate_columns=False,
        )

    if any(key in kwargs for key in ("sep", "delimiter", "header")):
        df = pd.read_csv(path, **kwargs)
        return _normalize_xyz_columns(df, normalize_xyz=normalize_xyz, file_path=path, format_name=format_name)

    parse_attempts: list[dict[str, object]] = []
    if default_sep is None:
        parse_attempts.append({})
    else:
        parse_attempts.append({"sep": default_sep})

    if fallback_sep is not None and fallback_sep != default_sep:
        parse_attempts.append({"sep": fallback_sep, "engine": "python"})

    last_exception: Exception | None = None
    for parse_kwargs in parse_attempts:
        try:
            df = pd.read_csv(path, **parse_kwargs)
        except Exception as e:
            last_exception = e
            continue

        lower_cols = {str(col).lower() for col in df.columns}
        if {"x", "y", "z"}.issubset(lower_cols):
            return _normalize_xyz_columns(df, normalize_xyz=normalize_xyz, file_path=path, format_name=format_name)

    headerless_kwargs: dict[str, object] = {"header": None}
    if fallback_sep is not None:
        headerless_kwargs.update({"sep": fallback_sep, "engine": "python"})
    elif default_sep is not None:
        headerless_kwargs["sep"] = default_sep

    try:
        df = pd.read_csv(path, **headerless_kwargs)
        return _normalize_xyz_columns(df, normalize_xyz=normalize_xyz, file_path=path, format_name=format_name)
    except Exception as e:
        if last_exception is not None:
            raise ValueError(
                f"Failed to parse {format_name} file '{path}'. "
                f"Last parse attempt failed with: {type(last_exception).__name__}: {last_exception}"
            ) from e
        raise


def write_delimited_coordinates(
    pointcloud: PointCloud,
    file_path: Path,
    *,
    header: bool,
    sep: str,
    columns: list[str] | None = None,
) -> None:
    data = pointcloud.data if columns is None else pointcloud.data[columns]
    data.to_csv(file_path, index=False, header=header, sep=sep)
