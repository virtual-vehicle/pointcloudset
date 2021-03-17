from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from lidar import Frame


def from_dataframe(df: pd.DataFrame) -> dict:
    return {"data": df}


def to_dataframe(frame: Frame) -> pd.DataFrame:
    return frame.data
