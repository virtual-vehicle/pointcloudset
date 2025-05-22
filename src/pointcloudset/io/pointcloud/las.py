from math import ceil, log10
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np

if TYPE_CHECKING:
    from pointcloudset import PointCloud

LAS_POINT_FORMAT = 7
LAS_VERSION = "1.4"
LAS_PRECISION = 0.000001  # m


def _choose_scale_offset(axis: np.ndarray) -> tuple[float, float]:
    """
    Return (scale, offset) guaranteeing int32 fit.
    PRECISION_HINT – user-desired resolution (e.g. 0.001 m for mm).  If None,
    we keep the finest resolution allowed by the int32 range, rounded to 10^n.
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
    """
    Return the smallest LAS data-type code ('u1', 'i1', …, 'f8') that can
    represent *arr* loss-lessly.

    LAS → NumPy mapping codes:
    u1/i1  : unsigned/signed   8-bit integer
    u2/i2  : unsigned/signed  16-bit integer
    u4/i4  : unsigned/signed  32-bit integer
    u8/i8  : unsigned/signed  64-bit integer
    f4/f8  : 32- / 64-bit float
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
    """
    Export a PointCloud to LAS/LAZ (point-format 7, LAS 1.4).
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
