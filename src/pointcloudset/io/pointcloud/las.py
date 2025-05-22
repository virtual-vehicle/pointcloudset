from math import ceil, log10
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np

if TYPE_CHECKING:
    from pointcloudset import PointCloud

LAS_POINT_FORMAT = 7
LAS_VERSION = "1.4"
LAS_PRECISION = 0.0001  # m

_NUMPY2LAS = {
    "uint8": "u1",
    "int8": "i1",
    "uint16": "u2",
    "int16": "i2",
    "uint32": "u4",
    "int32": "i4",
    "uint64": "u8",
    "int64": "i8",
    "float32": "f4",
    "float64": "f8",
}


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


def write_las(pointcloud: "PointCloud", file_path: Path) -> None:
    """
    Export to LAS/LAZ (point format 7).  Coordinates are stored at *precision_hint* or the
    finest safe resolution, whichever is coarser.
    Intensity is stored in the built-in PF-7 field, but all other fields are
    stored as ExtraBytes, see https://laspy.readthedocs.io/en/latest/intro.html
    Args:
        file_path (Path): Destination.
    Returns:
        None
    """
    df = pointcloud.data
    if not {"x", "y", "z"}.issubset(df.columns):
        raise ValueError("PointCloud must have x, y, z columns")

    header = laspy.LasHeader(point_format=LAS_POINT_FORMAT, version=LAS_VERSION)
    header.scales, header.offsets = zip(*(_choose_scale_offset(df[c].to_numpy()) for c in ("x", "y", "z")))
    las = laspy.LasData(header)

    # LAS coordinates
    las.x = df["x"].to_numpy()
    las.y = df["y"].to_numpy()
    las.z = df["z"].to_numpy()

    # LAS built-in point format 7 fields
    builtin = {n.lower() for n in las.point_format.dimension_names}
    for name in builtin - {"x", "y", "z"}:
        col = name if name in df.columns else None
        if col:
            setattr(las, name, df[col].to_numpy())

    #  LAS ExtraBytes for everything else
    extra = {c for c in df.columns if c.lower() not in builtin}
    for col in sorted(extra):
        las_type = _NUMPY2LAS.get(df[col].dtype.name, "f8")
        las.add_extra_dim(laspy.ExtraBytesParams(name=col, type=las_type))
        las[col] = df[col].to_numpy()

    las.write(Path(file_path))
