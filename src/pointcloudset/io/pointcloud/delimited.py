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


def read_delimited_coordinates(
    file_path: Path | str,
    *,
    format_name: str,
    default_sep: str | None,
    fallback_sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    path = Path(file_path)

    # If users explicitly provide column names, keep those names untouched.
    if "names" in kwargs:
        df = pd.read_csv(path, **kwargs)
        if {"x", "y", "z"}.issubset(set(df.columns)):
            return df
        raise ValueError(
            f"{format_name} file '{path}' was read with explicit column names, but columns x, y, z are required."
        )

    if any(key in kwargs for key in ("sep", "delimiter", "header")):
        return ensure_coordinate_columns(pd.read_csv(path, **kwargs), path, format_name)

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
            if {"x", "y", "z"}.issubset(set(df.columns)):
                return df
        except Exception as e:
            last_exception = e

    headerless_kwargs: dict[str, object] = {"header": None}
    if fallback_sep is not None:
        headerless_kwargs.update({"sep": fallback_sep, "engine": "python"})
    elif default_sep is not None:
        headerless_kwargs["sep"] = default_sep

    try:
        df = pd.read_csv(path, **headerless_kwargs)
        return ensure_coordinate_columns(df, path, format_name)
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
