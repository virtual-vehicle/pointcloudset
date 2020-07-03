import numpy as np
import open3d as o3d
import pandas as pd
import plotly
import pyntcloud
import pytest_check as check
import rospy
from pandas._testing import assert_frame_equal

from lidar import Frame


def test_init(testframe_mini_df: pd.DataFrame):
    frame = Frame(testframe_mini_df, rospy.rostime.Time(50))
    check.equal(type(frame), Frame)


def test_has_data(testframe_mini):
    check.equal(testframe_mini.has_data(), True)


def test_points(testframe_mini):
    points = testframe_mini.points.points
    check.equal(type(points), pd.DataFrame)
    check.equal(list(points.columns), ["x", "y", "z"])


def test_get_open3d_points(testframe_mini):
    pointcloud = testframe_mini.get_open3d_points()
    check.equal(type(pointcloud), o3d.open3d_pybind.geometry.PointCloud)
    check.equal(pointcloud.has_points(), True)


def test_measurments(testframe_mini):
    measurements = testframe_mini.measurments
    check.equal(type(measurements), pd.DataFrame)
    check.equal(
        list(measurements.columns),
        ["intensity", "t", "reflectivity", "ring", "noise", "range"],
    )


def test_timestamp(testframe_mini):
    check.equal(type(testframe_mini.timestamp), rospy.rostime.Time)


def test_len(testframe_mini: Frame):
    check.equal(type(len(testframe_mini)), int)
    check.equal(len(testframe_mini), 7)


def test_str(testframe_mini: Frame):
    check.equal(type(str(testframe_mini)), str)
    check.equal(
        str(testframe_mini),
        "pointcloud: with 7 points, data:['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range'], from Thursday, January 01, 1970 12:00:50",
    )


# test with actual data
def test_testframe_1_with_zero(testframe_withzero: Frame):
    check.equal(len(testframe_withzero), 131072)
    check.equal(testframe_withzero.has_data(), True)
    check.equal(testframe_withzero.timestamp.to_time(), 1592833242.7559116)


def test_testframe_1(testframe: Frame):
    check.equal(len(testframe), 45809)
    check.equal(testframe.has_data(), True)
    check.equal(testframe.timestamp.to_time(), 1592833242.7559116)


def test_testframe_index(testframe):
    check.equal(testframe.data.index[0], 0)
    check.equal(testframe.data.index[-1] + 1, len(testframe))
    check.equal(testframe.data.index.is_monotonic_increasing, True)


def test_testframe_2(testframe_mini: Frame):
    check.equal(len(testframe_mini), 7)
    check.equal(testframe_mini.has_data(), True)


def test_testframe_data(testframe: Frame):
    data = testframe.data
    check.equal(
        list(data.columns),
        ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"],
    )
    check.equal(data.shape, (45809, 9))
    assert_frame_equal(data, testframe.data)


def test_testframe_withzero_data(
    testframe_withzero: Frame, reference_data_with_zero_dataframe: pd.DataFrame
):
    data = testframe_withzero.data
    check.equal(
        list(data.columns),
        ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"],
    )
    check.equal(data.shape, (131072, 9))
    assert_frame_equal(data, reference_data_with_zero_dataframe)


def test_testframe_pointcloud(
    testframe_withzero: Frame, reference_pointcloud_withzero_dataframe: pd.DataFrame
):
    pointcloud = testframe_withzero.get_open3d_points()
    array = np.asarray(pointcloud.points)
    pointcloud_df = pd.DataFrame(array)
    sub_points = pointcloud.select_by_index(list(range(5000, 5550)))
    sub_array = np.asarray(sub_points.points)

    check.equal(pointcloud.is_empty(), False)
    check.equal(np.sum(sub_array), 108.0019814982079)
    check.equal(np.sum(array), 60573.190267673606)
    assert_frame_equal(pointcloud_df, reference_pointcloud_withzero_dataframe)


def test_distances_to_origin(testframe_mini):
    distances = testframe_mini.distances_to_origin()
    check.equal(type(distances), np.ndarray)
    check.equal(
        np.allclose(
            distances,
            np.asarray(
                [
                    0.0,
                    1.73205081,
                    937.15579489,
                    1058.09727232,
                    906.10064672,
                    991.83926827,
                    475.99837556,
                ]
            ),
        ),
        True,
    )


def test_plot1(testframe_mini: Frame):
    check.equal(
        type(testframe_mini.plot_interactive()), plotly.graph_objs._figure.Figure
    )
