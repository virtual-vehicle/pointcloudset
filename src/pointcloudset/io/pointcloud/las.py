from math import ceil, log10
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np

if TYPE_CHECKING:
    from pointcloudset import PointCloud

LAS_POINT_FORMAT = 7  # fixed
LAS_VERSION = "1.4"

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


def _choose_scale_offset(axis: np.ndarray, precision_hint: float | None = None) -> tuple[float, float]:
    """
    Return (scale, offset) guaranteeing int32 fit.
    *precision_hint* – user-desired resolution (e.g. 0.001 m for mm).  If None,
    we keep the finest resolution allowed by the int32 range, rounded to 10^n.
    """
    lo, hi = float(axis.min()), float(axis.max())
    offset = lo
    span = hi - lo

    if span == 0:  # flat axis
        return (precision_hint or 0.001), offset

    min_scale = span / (2**31 - 1)  # theoretical minimum
    rounded = 10 ** ceil(log10(min_scale))  # 10^n ≥ min_scale
    scale = max(rounded, precision_hint or 0)

    return scale, offset


def write_las(
    pointcloud: "PointCloud",
    file_path: Path,
    *,
    precision_hint: float | None = None,  # e.g. 0.001 for millimetres
) -> None:
    """
    Export to LAS/LAZ (PF-7).  Coordinates are stored at *precision_hint* or the
    finest safe resolution, whichever is coarser.  All remaining numeric
    columns become ExtraBytes unless already defined by PF-7.
    """
    df = pointcloud.data
    if not {"x", "y", "z"}.issubset(df.columns):
        raise ValueError("PointCloud must have x, y, z columns")

    # ---------- header ------------------------------------------------------
    hdr = laspy.LasHeader(point_format=LAS_POINT_FORMAT, version=LAS_VERSION)
    hdr.scales, hdr.offsets = zip(*(_choose_scale_offset(df[c].to_numpy(), precision_hint) for c in ("x", "y", "z")))
    las = laspy.LasData(hdr)

    # ---------- coordinates -------------------------------------------------
    las.x, las.y, las.z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()

    # ---------- built-in PF-7 fields ----------------------------------------
    builtin = {n.lower() for n in las.point_format.dimension_names}
    for name in builtin - {"x", "y", "z"}:
        col = name if name in df.columns else None
        if col:
            setattr(las, name, df[col].to_numpy())

    # ---------- ExtraBytes for everything else -----------------------------
    extra = {c for c in df.columns if c.lower() not in builtin}
    for col in sorted(extra):
        las_type = _NUMPY2LAS.get(df[col].dtype.name, "f8")
        las.add_extra_dim(laspy.ExtraBytesParams(name=col, type=las_type))
        las[col] = df[col].to_numpy()

    las.write(Path(file_path))
