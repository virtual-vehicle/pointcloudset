from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pointcloudset import Frame


def from_dataframe(df: pd.DataFrame) -> dict:
    """Converts pandas DataFrame to a Frame.

    Args:
        df (pandas.DataFrame): Pandas DataFrame to convert.

    Returns:
        dict: Returns data for Frame.
    """
    return {"data": df}


def to_dataframe(frame: Frame) -> pd.DataFrame:
    """Converts a Frame to a pandas DataFrame.

    Args:
        frame (Frame): Frame to convert.

    Returns:
        pandas.DataFrame: Data of Frame.
    """
    return frame.data
