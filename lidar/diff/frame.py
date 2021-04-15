"""
Frame differences.

Functions to calculate differences between frames.

"""

import numpy as np
import pandas as pd


def calculate_distance_to_frame(frame, target):
    """Calculate the point differences for each point which is also in the target fraem.
    Only points with the same orginal_id are compared. The results are added to the#
    data of the frame. (frame - target)

    Args:
        target (Frame): A Frame object to compute the differences.

    Raises:
        ValueError: If there are no points in FrameB with the same orginal_id
    """
    if not frame.has_original_id():
        raise ValueError("Frame does not contain original_id.")
    if not target.has_original_id():
        raise ValueError("Target does not contain original_id.")
    refrence_orginial_ids = frame.data.original_id.values
    target_orginal_ids = target.data.original_id.values
    intersection = np.intersect1d(refrence_orginial_ids, target_orginal_ids)
    if len(intersection) > 0:
        diff_list = [
            _calculate_single_point_difference(frame, target, id) for id in intersection
        ]
        orginal_types = [str(types) for types in diff_list[0].dtypes.values]
        target_type_dict = dict(zip(diff_list[0].columns.values, orginal_types))
        diff_df = pd.concat(diff_list)
        diff_df = diff_df.astype(target_type_dict)
        diff_df = diff_df.reset_index(drop=True)
        frame.data = frame.data.merge(diff_df, on="original_id", how="left")
        return frame
    else:
        raise ValueError("no intersection found between the frames.")


def _calculate_single_point_difference(frame, frameB, original_id: int) -> pd.DataFrame:
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
