from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import pytest_check as check
from pandas._testing import assert_frame_equal
from pyntcloud import PyntCloud

import lidar
from lidar import Frame


def test_from_pyntcloud(testlas1: Path):
    pyntcloud_data = PyntCloud.from_file(testlas1.as_posix())
    frame = lidar.Frame.from_instance("pyntcloud", pyntcloud_data)
    check.is_instance(frame, lidar.Frame)
    check.equal(frame.has_original_id, False)
    check.equal(len(list(frame.data.columns)), 13)


def test_to_pyntcloud(testframe_mini: Frame, testlas1: Path):
    pyntcloud_data = PyntCloud.from_file(testlas1.as_posix())
    res = testframe_mini.to_instance("pyntcloud")
    check.is_instance(res, PyntCloud)
    check.equal(
        list(res.points.columns.values), list(testframe_mini.data.columns.values)
    )


def test_to_open3d(testframe_mini: Frame):
    pointcloud = testframe_mini.to_instance("open3d")
    check.equal(type(pointcloud), o3d.cpu.pybind.geometry.PointCloud)
    check.equal(pointcloud.has_points(), True)
    check.equal(len(np.asarray(pointcloud.points)), len(testframe_mini))


def test_from_open3d(testframe_mini_real: Frame):
    open3d_data = testframe_mini_real.to_instance("open3d")
    frame = lidar.Frame.from_instance("open3d", open3d_data)
    test = frame.data - testframe_mini_real.data[["x", "y", "z"]]
    check.equal(set(list(test.max())).intersection([0.0, 0.0, 0.0]), {0.0})


def test_from_dataframe(testframe_mini_df: pd.DataFrame, testframe_mini: Frame):
    frame = lidar.Frame.from_instance("DataFrame", testframe_mini_df)
    check.is_instance(frame, Frame)
    test = frame.data - testframe_mini.data[["x", "y", "z"]]
    check.equal(set(list(test.max())).intersection([0.0, 0.0, 0.0]), {0.0})


def test_to_dataframe(testframe_mini_df: pd.DataFrame, testframe_mini: Frame):
    df = testframe_mini.to_instance("DataFrame")
    check.is_instance(df, pd.DataFrame)


def test_to_dataframe2(testframe_mini_df: pd.DataFrame, testframe_mini: Frame):
    df = testframe_mini.to_instance("pandas")
    check.is_instance(df, pd.DataFrame)