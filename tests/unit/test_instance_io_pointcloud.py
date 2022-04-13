from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import pytest_check as check
from pyntcloud import PyntCloud
import pytest

import pointcloudset
from pointcloudset import PointCloud


def test_from_pyntcloud(testlas1: Path):
    pyntcloud_data = PyntCloud.from_file(testlas1.as_posix())
    pointcloud = pointcloudset.PointCloud.from_instance("pyntcloud", pyntcloud_data)
    check.is_instance(pointcloud, pointcloudset.PointCloud)
    check.equal(pointcloud.has_original_id, False)
    check.equal(len(list(pointcloud.data.columns)), 13)


def test_to_pyntcloud(testpointcloud_mini: PointCloud, testlas1: Path):
    pyntcloud_data = PyntCloud.from_file(testlas1.as_posix())
    res = testpointcloud_mini.to_instance("pyntcloud")
    check.is_instance(res, PyntCloud)
    check.equal(
        list(res.points.columns.values), list(testpointcloud_mini.data.columns.values)
    )


def test_to_open3d(testpointcloud_mini: PointCloud):
    pointcloud = testpointcloud_mini.to_instance("open3d")
    check.equal(type(pointcloud), o3d.geometry.PointCloud)
    check.equal(pointcloud.dimension(), 3)
    check.equal(pointcloud.has_points(), True)
    check.equal(len(np.asarray(pointcloud.points)), len(testpointcloud_mini))


def test_from_open3d(testpointcloud_mini_real: PointCloud):
    open3d_data = testpointcloud_mini_real.to_instance("open3d")
    pointcloud = pointcloudset.PointCloud.from_instance("open3d", open3d_data)
    test = pointcloud.data - testpointcloud_mini_real.data[["x", "y", "z"]]
    check.equal(set(list(test.max())).intersection([0.0, 0.0, 0.0]), {0.0})


def test_from_dataframe(
    testpointcloud_mini_df: pd.DataFrame, testpointcloud_mini: PointCloud
):
    pointcloud = pointcloudset.PointCloud.from_instance(
        "DataFrame", testpointcloud_mini_df
    )
    check.is_instance(pointcloud, PointCloud)
    test = pointcloud.data - testpointcloud_mini.data[["x", "y", "z"]]
    check.equal(set(list(test.max())).intersection([0.0, 0.0, 0.0]), {0.0})


@pytest.mark.paramterize("to", ["DataFrame", "dataframe", "pandas", "Pandas"])
def test_to_dataframe(testpointcloud_mini: PointCloud):
    df = testpointcloud_mini.to_instance("DataFrame")
    check.is_instance(df, pd.DataFrame)
