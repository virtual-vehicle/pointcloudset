from pathlib import Path

import pandas as pd
import pytest
import pytest_check as check

import pointcloudset
from pointcloudset import PointCloud


def test_from_instance_unsupported_library(testpointcloud_mini_df: pd.DataFrame):
    with pytest.raises(ValueError):
        pointcloudset.PointCloud.from_instance("pyntcloud", testpointcloud_mini_df)


def test_to_instance_unsupported_library(testpointcloud_mini: PointCloud):
    with pytest.raises(ValueError):
        testpointcloud_mini.to_instance("pyntcloud")


def test_from_dataframe(testpointcloud_mini_df: pd.DataFrame, testpointcloud_mini: PointCloud):
    pointcloud = pointcloudset.PointCloud.from_instance("DataFrame", testpointcloud_mini_df)
    check.is_instance(pointcloud, PointCloud)
    test = pointcloud.data - testpointcloud_mini.data[["x", "y", "z"]]
    check.equal(set(list(test.max())).intersection([0.0, 0.0, 0.0]), {0.0})


@pytest.mark.paramterize("to", ["DataFrame", "dataframe", "pandas", "Pandas"])
def test_to_dataframe(testpointcloud_mini: PointCloud):
    df = testpointcloud_mini.to_instance("DataFrame")
    check.is_instance(df, pd.DataFrame)
