import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pointcloudset import Dataset, PointCloud


@pytest.fixture()
def testdata_path() -> Path:
    return Path(__file__).parent.absolute() / "testdata"


@pytest.fixture()
def testbag1():
    return Path(__file__).parent.absolute() / "testdata/test.bag"


@pytest.fixture()
def testlas1():
    return Path(__file__).parent.absolute() / "testdata/diamond.las"


@pytest.fixture()
def testset(testbag1):
    return Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=False)


@pytest.fixture()
def testset_withzero(testbag1):
    return Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)


@pytest.fixture()
def testpointcloud(testset):
    return testset[1]


@pytest.fixture()
def testframe0(testset):
    return testset[0]


@pytest.fixture()
def testpointcloud_withzero(testset_withzero):
    return testset_withzero[1]


@pytest.fixture()
def testpointcloud_mini_df():
    columns = [
        "x",
        "y",
        "z",
        "intensity",
        "t",
        "reflectivity",
        "ring",
        "noise",
        "range",
    ]
    np.random.seed(5)
    df1 = pd.DataFrame(np.zeros(shape=(1, len(columns))), columns=columns)
    df2 = pd.DataFrame(np.ones(shape=(1, len(columns))), columns=columns)
    df3 = pd.DataFrame(-1.0 * np.ones(shape=(1, len(columns))), columns=columns)
    df4 = pd.DataFrame(
        np.random.randint(0, 1000, size=(5, len(columns))) * np.random.random(),
        columns=columns,
    )
    return pd.concat([df1, df2, df3, df4]).reset_index(drop=True)


@pytest.fixture()
def reference_data_with_zero_dataframe():
    filename = (
        Path(__file__).parent.absolute()
        / "testdata/testpointcloud_withzero_dataframe.pkl"
    )
    return pd.read_pickle(filename)


@pytest.fixture()
def reference_pointcloud_withzero_dataframe():
    filename = (
        Path(__file__).parent.absolute()
        / "testdata/testpointcloud_withzero_pointcloud.pkl"
    )
    return pd.read_pickle(filename)


@pytest.fixture()
def testpointcloud_mini(testpointcloud_mini_df) -> PointCloud:
    return PointCloud(
        data=testpointcloud_mini_df,
        timestamp=datetime.datetime(2020, 1, 1),
        orig_file="/fake/testrame_mini.bag",
    )


@pytest.fixture()
def testpointcloud_mini_real(testpointcloud) -> PointCloud:
    return (
        testpointcloud.limit("x", -1, 1)
        .limit("y", -1, 1)
        .limit("z", -1, 1)
        .limit("intensity", 0, 10)
    )


@pytest.fixture()
def testpointcloud_mini_real_plus1(testpointcloud_mini_real) -> PointCloud:
    testdata = testpointcloud_mini_real.data.copy(deep=True)
    testdata = testdata + 1.0
    testdata["original_id"] = testpointcloud_mini_real.data["original_id"]
    return PointCloud(data=testdata)


@pytest.fixture()
def testpointcloud_mini_real_other_original_id(testpointcloud_mini_real) -> PointCloud:
    testdata = testpointcloud_mini_real.data.copy(deep=True)
    testdata["original_id"] = testdata["original_id"] + 1000000
    return PointCloud(data=testdata)


@pytest.fixture()
def testdataset_mini_real(
    testpointcloud_mini_real, testpointcloud_mini_real_plus1
) -> Dataset:
    pointclouds = [testpointcloud_mini_real, testpointcloud_mini_real_plus1]
    return Dataset.from_instance("POINTCLOUDS", pointclouds)


@pytest.fixture()
def testdataset_mini_same(testpointcloud_mini_real) -> Dataset:
    pointclouds = [testpointcloud_mini_real, testpointcloud_mini_real]
    return Dataset.from_instance("POINTCLOUDS", pointclouds)


@pytest.fixture()
def test_kitti() -> Dataset:
    filename = (
        Path(__file__).parent.absolute()
        / "testdata/kitti_velodyne/kitti_2011_09_26_drive_0002_synce"
    )
    return Dataset.from_file(filename)
