import pandas as pd
import numpy as np


def calculate_single_point_difference(frame, frameB, original_id: int) -> pd.DataFrame:
    """Calculate the difference of one element of a Point in the current Frame to
    the correspoing point in Frame B. Both frames must contain the same orginal_id.

    Args:
        frameB (Frame): Frame which contains the point to comapare to.
        original_id (int): Orginal ID of the point.

    Returns:
        pd.DataFrame: A single row DataFrame with the differences (A - B).
    """
    pointA = frame.extract_point(original_id, use_orginal_id=True)
    try:
        pointB = frameB.extract_point(original_id, use_orginal_id=True)
        difference = pointA - pointB
    except IndexError:
        # there is no point with the orignal_id in frameB
        difference = pointA
        difference.loc[:] = np.nan
    difference = difference.drop(["original_id"], axis=1)
    difference.columns = [f"{column} difference" for column in difference.columns]
    difference["original_id"] = pointA["original_id"]
    return difference
