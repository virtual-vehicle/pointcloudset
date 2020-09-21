from pathlib import Path
import IPython
import numpy as np
import open3d as o3d
import pandas as pd
import plotly
import pytest
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
    check.equal(len(np.asarray(pointcloud.points)), len(testframe_mini))
    testlist = list(np.asarray(pointcloud.points))
    print(testlist)


def test_measurments(testframe_mini):
    measurements = testframe_mini.measurments
    check.equal(type(measurements), pd.DataFrame)


def test_timestamp(testframe_mini):
    check.equal(type(testframe_mini.timestamp), rospy.rostime.Time)


def test_org_file(testframe):
    check.equal(type(testframe.orig_file), str)
    check.equal(Path(testframe.orig_file).stem, "test")


def test_len(testframe_mini: Frame):
    check.equal(type(len(testframe_mini)), int)
    check.equal(len(testframe_mini), 7)


def test_str(testframe: Frame):
    check.equal(type(str(testframe)), str)
    check.equal(
        str(testframe),
        "pointcloud: with 45809 points, data:['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range'], from Monday, June 22, 2020 01:40:42",
    )


def test_add_column(testframe_mini: Frame):
    testframe_mini.add_column("test", testframe_mini.data["x"])
    after_columns = list(testframe_mini.data.columns.values)
    check.equal(
        str(after_columns),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'test']",
    )


def test_calculate_distance_to_plane1(testframe_mini: Frame):
    testframe_mini.calculate_distance_to_plane(
        plane_model=np.array([1, 0, 0, 0]), absolute_values=False
    )
    check.equal(
        str(list(testframe_mini.data.columns.values)),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'distance to plane: [1,0,0,0]']",
    )
    check.equal(testframe_mini.data["distance to plane: [1,0,0,0]"][1], 1.0)


def test_calculate_distance_to_plane2(testframe_mini: Frame):
    testframe_mini.calculate_distance_to_plane(
        plane_model=np.array([-1, 0, 0, 0]), absolute_values=False
    )
    check.equal(
        str(list(testframe_mini.data.columns.values)),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'distance to plane: [-1, 0, 0, 0]']",
    )
    check.equal(testframe_mini.data["distance to plane: [-1, 0, 0, 0]"][1], -1.0)


def test_calculate_distance_to_plane3(testframe_mini: Frame):
    testframe_mini.calculate_distance_to_plane(
        plane_model=np.array([-1, 0, 0, 0]), absolute_values=True
    )
    check.equal(testframe_mini.data["distance to plane: [-1, 0, 0, 0]"][1], 1.0)


def test_describe(testframe: Frame):
    check.equal(type(testframe.describe()), pd.DataFrame)


# test with actual data
def test_testframe_1_with_zero(testframe_withzero: Frame):
    check.equal(len(testframe_withzero), 131072)
    check.equal(testframe_withzero.has_data(), True)
    check.equal(testframe_withzero.timestamp.to_time(), 1592833242.7559116)


def test_testframe_1(testframe: Frame):
    # testframe.data.to_pickle("/workspaces/lidar/tests/testdata/testframe_dataframe.pkl")
    check.equal(len(testframe), 45809)
    check.equal(testframe.has_data(), True)
    check.equal(testframe.timestamp.to_time(), 1592833242.7559116)
    check.equal(len(testframe), 45809)


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
    pointcloud = testframe_withzero.get_open3d_points()
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


def test_plot2(testframe_mini: Frame):
    check.equal(
        type(testframe_mini.plot_interactive(backend="pyntcloud")),
        plotly.graph_objs._figure.Figure,
    )


def test_plot2(testframe_mini: Frame):
    check.equal(
        type(testframe_mini.plot_interactive(backend="pyntcloud")),
        IPython.lib.display.IFrame,
    )


def test_plot_error(testframe_mini: Frame):
    with pytest.raises(ValueError):
        testframe_mini.plot_interactive(backend="fake")


def test_plot_overlay(testframe: Frame):
    smaller = testframe.limit("x", -0.5, 0.0)
    smaller2 = testframe.limit("x", -0.1, 0.0)
    check.equal(
        type(smaller.plot_overlay({"Smaller2": smaller2})),
        plotly.graph_objs._figure.Figure,
    )


def test_to_csv(testframe: Frame, tmp_path: Path):
    testfile_name = tmp_path.joinpath("test_timestamp_1592833242755911566.csv")
    testframe.to_csv(testfile_name)
    print(testfile_name)
    print(testfile_name.exists())
    check.equal(testfile_name.exists(), True)
    read_frame = pd.read_csv(testfile_name)
    test_values = read_frame.iloc[0].values
    np.testing.assert_allclose(
        [
            1.4383683e00,
            -4.0477440e-01,
            2.1055990e-01,
            1.1000000e01,
            +3.5151600e06,
            2.0000000e00,
            1.6000000e01,
            3.5000000e01,
            +1.5090000e03,
        ],
        test_values,
        rtol=1e-10,
        atol=0,
    )
