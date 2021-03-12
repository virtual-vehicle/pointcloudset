from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rospy

from lidar import Dataset, Frame


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
def testframe(testset):
    return testset[1]


@pytest.fixture()
def testframe0(testset):
    return testset[0]


@pytest.fixture()
def testframe_withzero(testset_withzero):
    return testset_withzero[1]


@pytest.fixture()
def testframe_mini_df():
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
        Path(__file__).parent.absolute() / "testdata/testframe_withzero_dataframe.pkl"
    )
    return pd.read_pickle(filename)


@pytest.fixture()
def reference_pointcloud_withzero_dataframe():
    filename = (
        Path(__file__).parent.absolute() / "testdata/testframe_withzero_pointcloud.pkl"
    )
    return pd.read_pickle(filename)


@pytest.fixture()
def testframe_mini(testframe_mini_df) -> Frame:
    return Frame(
        data=testframe_mini_df,
        timestamp=rospy.rostime.Time(50),
        orig_file="/fake/testrame_mini.bag",
    )


@pytest.fixture()
def testframe_mini_real(testframe) -> Frame:
    return (
        testframe.limit("x", -1, 1)
        .limit("y", -1, 1)
        .limit("z", -1, 1)
        .limit("intensity", 0, 10)
    )


@pytest.fixture()
def testframe_mini_real_plus1(testframe_mini_real) -> Frame:
    testdata = testframe_mini_real.data.copy(deep=True)
    testdata = testdata + 1.0
    testdata["original_id"] = testframe_mini_real.data["original_id"]
    return Frame(data=testdata)


@pytest.fixture()
def testframe_mini_real_other_original_id(testframe_mini_real) -> Frame:
    testdata = testframe_mini_real.data.copy(deep=True)
    testdata["original_id"] = testdata["original_id"] + 1000000
    return Frame(data=testdata)
