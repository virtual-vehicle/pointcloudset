from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import pytest
import pytest_check as check
import rospy
from pandas._testing import assert_frame_equal
from pyntcloud import PyntCloud
from datetime import datetime

from lidar import Frame


def test_init(testframe_mini_df: pd.DataFrame):
    frame = Frame(testframe_mini_df, rospy.rostime.Time(50))
    check.equal(type(frame), Frame)


def test_has_data(testframe_mini: Frame):
    check.equal(testframe_mini._has_data(), True)


def test_has_original_id(testframe_mini: Frame):
    check.equal(testframe_mini.has_original_id(), False)


def test_has_original_id2(testframe_mini_real: Frame):
    check.equal(testframe_mini_real.has_original_id(), True)


def test_contains_original_id_number(testframe: Frame):
    check.equal(testframe._contains_original_id_number(4700), True)
    check.equal(testframe._contains_original_id_number(100000000), False)
    check.equal(testframe._contains_original_id_number(1), False)
    check.equal(testframe._contains_original_id_number(0), False)
    check.equal(testframe._contains_original_id_number(-1000), False)


def test_points(testframe_mini):
    points = testframe_mini.points
    check.is_instance(points, PyntCloud)


def test_points2(testframe_mini):
    points = testframe_mini.points
    check.equal(
        list(points.points.columns),
        ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"],
    )


def test_timestamp(testframe_mini):
    check.equal(type(testframe_mini.timestamp), datetime)


def test_org_file(testframe):
    check.equal(type(testframe.orig_file), str)
    check.equal(Path(testframe.orig_file).stem, "test")


def test_len(testframe_mini: Frame):
    check.equal(type(len(testframe_mini)), int)
    check.equal(len(testframe_mini), 8)


def test_getitem_single(testframe: Frame):
    extracted = testframe[0]
    check.equal(type(extracted), pd.DataFrame)
    check.equal(len(extracted), 1)
    check.equal(extracted.shape, (1, 10))
    check.equal(type(extracted.original_id.values[0]), np.uint32)
    check.equal(extracted.original_id.values[0], 4624)


def test_getitem_single_error(testframe: Frame):
    with pytest.raises(IndexError):
        testframe[1000000000]


def test_getitem_slice(testframe: Frame):
    extracted = testframe[0:3]
    check.equal(type(extracted), pd.DataFrame)
    check.equal(len(extracted), 3)
    check.equal(type(extracted.original_id.values[0]), np.uint32)
    check.equal(
        (extracted.original_id.values == np.array([4624, 4688, 4692])).all(), True
    )


def test_getitem_slice_full(testframe: Frame):
    extracted = testframe[:]
    check.equal(len(extracted), len(testframe))
    check.equal(testframe.data.equals(extracted), True)


def test_getitem_slice_step(testframe: Frame):
    extracted = testframe[0:10:2]
    check.equal(type(extracted), pd.DataFrame)
    check.equal(len(extracted), 5)
    check.equal(
        (
            extracted.original_id.values == np.array([4624, 4692, 4700, 4752, 4760])
        ).all(),
        True,
    )


def test_str(testframe: Frame):
    check.equal(type(str(testframe)), str)
    check.equal(
        str(testframe),
        "pointcloud: with 45809 points, data:['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'original_id'], from Monday, June 22, 2020 01:40:42",
    )


def test_repr(testframe_mini: Frame):
    check.equal(type(repr(testframe_mini)), str)
    check.equal(len(repr(testframe_mini)), 785)


def test_add_column(testframe_mini: Frame):
    newframe = testframe_mini._add_column("test", testframe_mini.data["x"])
    check.equal(type(newframe), Frame)
    after_columns = list(testframe_mini.data.columns.values)
    check.equal(
        str(after_columns),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'test']",
    )


def test_describe(testframe: Frame):
    check.equal(type(testframe.describe()), pd.DataFrame)


# test with actual data
def test_testframe_1_with_zero(testframe_withzero: Frame):
    check.equal(len(testframe_withzero), 131072)
    check.equal(testframe_withzero._has_data(), True)


def test_testframe_1(testframe: Frame):
    # testframe.data.to_pickle("/workspaces/lidar/tests/testdata/testframe_dataframe.pkl")
    check.equal(len(testframe), 45809)
    check.equal(testframe._has_data(), True)
    check.equal(len(testframe), 45809)


def test_testframe_index(testframe):
    check.equal(testframe.data.index[0], 0)
    check.equal(testframe.data.index[-1] + 1, len(testframe))
    check.equal(testframe.data.index.is_monotonic_increasing, True)


def test_testframe_2(testframe_mini: Frame):
    check.equal(len(testframe_mini), 8)
    check.equal(testframe_mini._has_data(), True)


def test_testframe_data(testframe: Frame):
    data = testframe.data
    check.equal(
        list(data.columns),
        [
            "x",
            "y",
            "z",
            "intensity",
            "t",
            "reflectivity",
            "ring",
            "noise",
            "range",
            "original_id",
        ],
    )
    check.equal(data.shape, (45809, 10))


def test_testframe_data_types(testframe: Frame):
    types = [str(types) for types in testframe.data.dtypes.values]
    check.equal(
        types,
        [
            "float32",
            "float32",
            "float32",
            "float32",
            "uint32",
            "uint16",
            "uint8",
            "uint16",
            "uint32",
            "uint32",
        ],
    )


def test_testframe_withzero_data(
    testframe_withzero: Frame, reference_data_with_zero_dataframe: pd.DataFrame
):
    data = testframe_withzero.data
    # data.to_pickle("/workspaces/lidar/tests/testdata/testframe_withzero_dataframe.pkl")
    check.equal(
        list(data.columns),
        ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"],
    )
    check.equal(data.shape, (131072, 9))
    assert_frame_equal(data, reference_data_with_zero_dataframe)


def test_testframe_pointcloud(
    testframe_withzero: Frame, reference_pointcloud_withzero_dataframe: pd.DataFrame
):
    pointcloud = testframe_withzero.to_instance("open3d")
    array = np.asarray(pointcloud.points)
    pointcloud_df = pd.DataFrame(array)
    # pointcloud_df.to_pickle(
    #    "/workspaces/lidar/tests/testdata/testframe_withzero_pointcloud.pkl"
    # )
    sub_points = pointcloud.select_by_index(list(range(5000, 5550)))
    sub_array = np.asarray(sub_points.points)

    check.equal(pointcloud.is_empty(), False)
    check.equal(np.sum(sub_array), 108.0019814982079)
    check.equal(np.sum(array), 60573.190267673606)
    assert_frame_equal(pointcloud_df, reference_pointcloud_withzero_dataframe)
