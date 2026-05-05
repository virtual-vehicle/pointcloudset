from math import ceil, log10
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pointcloudset import PointCloud

LAS_POINT_FORMAT = 7
LAS_VERSION = "1.4"
LAS_PRECISION = 0.000001  # m


def read_las(file_path: Path | str, normalize_xyz: bool = False) -> pd.DataFrame:
    """Read a LAS/LAZ pointcloud file into a dataframe.

    Args:
        file_path: Path to the LAS/LAZ file.
        normalize_xyz: Whether to convert LAS-native ``X/Y/Z`` coordinates to
            lowercase ``x/y/z`` for pointcloudset internals.

    Returns:
        A dataframe containing pointcloud columns.

    Raises:
        ValueError: If ``normalize_xyz`` is not enabled.
    """
    path = Path(file_path)
    if not normalize_xyz:
        raise ValueError(
            f"LAS file '{path}' stores coordinates as X/Y/Z. "
            "pointcloudset expects lowercase x/y/z internally. "
            "Pass normalize_xyz=True to convert X/Y/Z to x/y/z."
        )

    las = laspy.read(path)
    raw = las.points.array

    data = {
        "x": np.asarray(las.x),
        "y": np.asarray(las.y),
        "z": np.asarray(las.z),
    }
    for name in raw.dtype.names:
        if name in {"X", "Y", "Z"}:
            continue
        data[name] = raw[name]

    return pd.DataFrame(data)


def _choose_scale_offset(axis: np.ndarray) -> tuple[float, float]:
    """Compute LAS scale and offset for a coordinate axis.

    Args:
        axis: Coordinate values for one axis.

    Returns:
        A ``(scale, offset)`` tuple that keeps integer storage within int32
        limits while respecting minimum precision.
    """
    lo, hi = float(axis.min()), float(axis.max())
    offset = lo
    span = hi - lo

    if span == 0:  # flat axis
        return (LAS_PRECISION), offset

    min_scale = span / (2**31 - 1)  # theoretical minimum
    rounded = 10 ** ceil(log10(min_scale))  # 10^n ≥ min_scale
    scale = max(rounded, LAS_PRECISION or 0)

    return scale, offset


def _best_las_type(arr: np.ndarray) -> str:
    """Select the smallest compatible LAS scalar type for array values.

    Args:
        arr: NumPy array with values for one output field.

    Returns:
        LAS type code such as ``u1``, ``i4``, ``f4``, or ``f8``.
    """
    dt = arr.dtype

    # Integers
    if dt.kind in {"u", "i"}:
        lo, hi = int(arr.min()), int(arr.max())
        if dt.kind == "u":  # unsigned
            if hi <= np.iinfo(np.uint8).max:
                return "u1"
            if hi <= np.iinfo(np.uint16).max:
                return "u2"
            if hi <= np.iinfo(np.uint32).max:
                return "u4"
            return "u8"
        else:  # signed
            if lo >= np.iinfo(np.int8).min and hi <= np.iinfo(np.int8).max:
                return "i1"
            if lo >= np.iinfo(np.int16).min and hi <= np.iinfo(np.int16).max:
                return "i2"
            if lo >= np.iinfo(np.int32).min and hi <= np.iinfo(np.int32).max:
                return "i4"
            return "i8"

    # Floats
    if dt.kind == "f":
        # If nothing is lost when casting to f4, prefer it.
        if np.allclose(arr.astype(np.float32), arr, rtol=0, atol=0):
            return "f4"
        return "f8"

    #  Booleans / bit-fields
    if dt.kind == "b":
        return "u1"  # 0 / 1

    # default to 64-bit float
    return "f8"


def write_las(pointcloud: "PointCloud", file_path: Path) -> None:
    """Write a pointcloud to LAS/LAZ using point format 7 and LAS 1.4.

    Args:
        pointcloud: PointCloud instance to serialize.
        file_path: Destination LAS/LAZ file path.

    Raises:
        ValueError: If the pointcloud is missing ``x``, ``y``, or ``z``.
    """
    df = pointcloud.data
    if not {"x", "y", "z"}.issubset(df.columns):
        raise ValueError("PointCloud must have x, y, z columns")

    # ── header ----------------------------------------------------------------
    header = laspy.LasHeader(point_format=LAS_POINT_FORMAT, version=LAS_VERSION)
    header.scales, header.offsets = zip(*(_choose_scale_offset(df[c].to_numpy()) for c in ("x", "y", "z")))
    las = laspy.LasData(header)

    # ── coordinates -----------------------------------------------------------
    las.x, las.y, las.z = (df[c].to_numpy() for c in ("x", "y", "z"))

    # ── built-in PF-7 dimensions ---------------------------------------------
    builtin = {n.lower() for n in las.point_format.dimension_names}
    builtin.update(
        {
            "bit_fields",
            "classification_flags",
            "return_number",
            "number_of_returns",
            "scan_direction_flag",
            "edge_of_flight_line",
        }
    )

    for name in builtin - {"x", "y", "z"}:
        if name in df.columns:
            setattr(las, name, df[name].to_numpy())

    # ── ExtraBytes for user columns ------------------------------------------
    extra_cols = [c for c in df.columns if c.lower() not in builtin]
    for col in sorted(extra_cols):
        las_type = _best_las_type(df[col].to_numpy())
        las.add_extra_dim(laspy.ExtraBytesParams(name=col, type=las_type))
        las[col] = df[col].to_numpy()

    las.write(Path(file_path))
