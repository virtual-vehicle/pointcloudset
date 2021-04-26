"""
Functions to calculate differences between frames.
"""

import numpy as np
import pandas as pd
import lidar


def calculate_distance_to_frame(frame, target):
    """Calculate the differences for each point in a frame which also exists in
    the target frame (frame - target). Points with the same original_id are compared.

    Note:
        Adds the results to the data of the frame.

    Args:
        frame (Frame): Frame for which the differences to the target are calculated.
        target (Frame): Frame which is substracted from frame.

    Returns:
        Frame: Frame including differences to target.

    Raises:
        ValueError: If there are no original_ids in frame or in target or if they have
            no common original_ids.
        NotImplementedError: If difference has already been calculated.
    """
    if "x difference" in frame.data.columns:
        raise NotImplementedError("Differences of differences are not supported.")
    if not frame.has_original_id:
        raise ValueError("Frame does not contain original_id.")
    if not target.has_original_id:
        raise ValueError("Target does not contain original_id.")
    reference_original_ids = frame.data.original_id.values
    target_original_ids = target.data.original_id.values
    intersection = np.intersect1d(reference_original_ids, target_original_ids)
    if len(intersection) > 0:
        return _calculate_difference(intersection, frame, target)

    else:
        raise ValueError("No common original_ids in frame and target.")


def _calculate_difference(intersection: np.ndarray, frame, target):
    diff_list = [
        _calculate_single_point_difference(frame, target, id) for id in intersection
    ]
    original_types = [str(types) for types in diff_list[0].dtypes.values]
    target_type_dict = dict(zip(diff_list[0].columns.values, original_types))
    diff_df = pd.concat(diff_list)
    diff_df = diff_df.astype(target_type_dict)
    diff_df = diff_df.reset_index(drop=True)
    frame.data = frame.data.merge(diff_df, on="original_id", how="left")
    return frame


def _calculate_single_point_difference(frame, frameB, original_id: int) -> pd.DataFrame:
    """Calculate the difference of one element of a Point in the current Frame to
    the corresponding point in Frame B. Both frames must contain the same original_id.

    Args:
        frameB (Frame): Frame which contains the point to comapare to.
        original_id (int): Original ID of the point.

    Returns:
        pd.DataFrame: A single row DataFrame with the differences (A - B).
    """
    pointA = frame.extract_point(original_id, use_original_id=True)
    try:
        pointB = frameB.extract_point(original_id, use_original_id=True)
        difference = pointA - pointB
    except IndexError:
        # there is no point with the orignal_id in frameB
        difference = pointA
        difference.loc[:] = np.nan
    difference = difference.drop(["original_id"], axis=1)
    difference.columns = [f"{column} difference" for column in difference.columns]
    difference["original_id"] = pointA["original_id"]
    return difference
