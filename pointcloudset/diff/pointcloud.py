"""
Functions to calculate differences between frames.
"""

import numpy as np
import pandas as pd

import pointcloudset


def calculate_distance_to_pointcloud(pointcloud, target):
    """Calculate the differences for each point in a pointcloud which also exists in
    the target pointcloud (pointcloud - target). Points with the same original_id are compared.

    Note:
        Adds the results to the data of the pointcloud.

    Args:
        pointcloud (PointCloud): PointCloud for which the differences to the target are calculated.
        target (PointCloud): PointCloud which is substracted from pointcloud.

    Returns:
        PointCloud: PointCloud including differences to target.

    Raises:
        ValueError: If there are no original_ids in pointcloud or in target or if they have
            no common original_ids.
        NotImplementedError: If difference has already been calculated.
    """
    if "x difference" in pointcloud.data.columns:
        raise NotImplementedError("Differences of differences are not supported.")
    if not pointcloud.has_original_id:
        raise ValueError("PointCloud does not contain original_id.")
    if not target.has_original_id:
        raise ValueError("Target does not contain original_id.")
    reference_original_ids = pointcloud.data.original_id.values
    target_original_ids = target.data.original_id.values
    intersection = np.intersect1d(reference_original_ids, target_original_ids)
    if len(intersection) > 0:
        return _calculate_difference(intersection, pointcloud, target)

    else:
        raise ValueError("No common original_ids in pointcloud and target.")


def _calculate_difference(intersection: np.ndarray, pointcloud, target):
    diff_list = [
        _calculate_single_point_difference(pointcloud, target, id)
        for id in intersection
    ]
    original_types = [str(types) for types in diff_list[0].dtypes.values]
    target_type_dict = dict(zip(diff_list[0].columns.values, original_types))
    diff_df = pd.concat(diff_list)
    diff_df = diff_df.astype(target_type_dict)
    diff_df = diff_df.reset_index(drop=True)
    pointcloud.data = pointcloud.data.merge(diff_df, on="original_id", how="left")
    return pointcloud


def _calculate_single_point_difference(
    pointcloud, frameB, original_id: int
) -> pd.DataFrame:
    """Calculate the difference of one element of a Point in the current PointCloud to
    the corresponding point in PointCloud B. Both frames must contain the same original_id.

    Args:
        frameB (PointCloud): PointCloud which contains the point to comapare to.
        original_id (int): Original ID of the point.

    Returns:
        pd.DataFrame: A single row DataFrame with the differences (A - B).
    """
    pointA = pointcloud.extract_point(original_id, use_original_id=True)
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
