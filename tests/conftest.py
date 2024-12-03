import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pointcloudset import Dataset, PointCloud

ROS1FILE = Path(__file__).parent.absolute() / "testdata/test.bag"
ROS2FILE = Path(__file__).parent.absolute() / "testdata/ros2"
ROS2MCAPFILE = Path(__file__).parent.absolute() / "testdata/ros2_mcap"
ROS2MCAPFILE_LIVOX = Path(__file__).parent.absolute() / "testdata/ros2_mcap_livox"


@pytest.fixture()
def testdata_path() -> Path:
    return Path(__file__).parent.absolute() / "testdata"


@pytest.fixture()
def testdata_path_large() -> Path:
    return Path(__file__).parent.absolute() / "testdata/large"


@pytest.fixture()
def testbag1():
    return ROS1FILE


@pytest.fixture()
def testros2():
    return ROS2FILE


@pytest.fixture()
def testros2mcap():
    return ROS2MCAPFILE


@pytest.fixture()
def testros2mcap_livox():
    return ROS2MCAPFILE_LIVOX


@pytest.fixture()
def testlas1():
    return Path(__file__).parent.absolute() / "testdata/las_files/diamond.las"


@pytest.fixture()
def testlasvz6000_1():
    return Path(__file__).parent.absolute() / "testdata/las_files/VZ6000_1.las"


@pytest.fixture()
def testlasvz6000_2():
    return Path(__file__).parent.absolute() / "testdata/las_files/VZ6000_2.las"


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
    filename = Path(__file__).parent.absolute() / "testdata/testpointcloud_withzero_dataframe.pkl"
    return pd.read_pickle(filename)


@pytest.fixture()
def reference_pointcloud_withzero_dataframe():
    filename = Path(__file__).parent.absolute() / "testdata/testpointcloud_withzero_pointcloud.pkl"
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
    testpointcloud_mini_real = testpointcloud.limit("x", -1, 1).limit("y", -1, 1).limit("intensity", 0, 10)
    testpointcloud_mini_real.timestamp = datetime.datetime(2020, 1, 1, 0, 0)
    return testpointcloud_mini_real


@pytest.fixture()
def testpointcloud_mini_real_later(testpointcloud) -> PointCloud:
    testpointcloud_mini_real_later = testpointcloud.limit("x", -1, 1).limit("y", -1, 1).limit("intensity", 0, 10)
    testpointcloud_mini_real_later.timestamp = datetime.datetime(2020, 1, 1, 0, 1)
    return testpointcloud_mini_real_later


@pytest.fixture()
def testpointcloud_mini_real_plus1(testpointcloud_mini_real_later) -> PointCloud:
    testdata = testpointcloud_mini_real_later.data.copy(deep=True)
    testdata = testdata + 1.0
    testdata["original_id"] = testpointcloud_mini_real_later.data["original_id"]
    return PointCloud(data=testdata, timestamp=datetime.datetime(2020, 1, 1, 0, 1))


@pytest.fixture()
def testpointcloud_mini_real_other_original_id(testpointcloud_mini_real) -> PointCloud:
    testdata = testpointcloud_mini_real.data.copy(deep=True)
    testdata["original_id"] = testdata["original_id"] + 1000000
    return PointCloud(data=testdata)


@pytest.fixture()
def testdataset_mini_real(testpointcloud_mini_real, testpointcloud_mini_real_plus1) -> Dataset:
    pointclouds = [testpointcloud_mini_real, testpointcloud_mini_real_plus1]
    return Dataset.from_instance("POINTCLOUDS", pointclouds)


@pytest.fixture()
def testdataset_mini_same(testpointcloud_mini_real, testpointcloud_mini_real_later) -> Dataset:
    pointclouds = [testpointcloud_mini_real, testpointcloud_mini_real_later]
    return Dataset.from_instance("POINTCLOUDS", pointclouds)


@pytest.fixture()
def test_kitti() -> Dataset:
    filename = Path(__file__).parent.absolute() / "testdata/kitti_velodyne/kitti_2011_09_26_drive_0002_synce"
    return Dataset.from_file(filename)


@pytest.fixture()
def testdataset_with_empty_frame(testdataset_mini_real: Dataset):
    def pipeline(pointcloud: PointCloud) -> PointCloud:
        return pointcloud.filter("value", "x", "<", 0)

    return testdataset_mini_real.apply(pipeline)


@pytest.fixture()
def testvz6000_1(testlasvz6000_1):
    return PointCloud.from_file(testlasvz6000_1, timestamp=datetime.datetime(2022, 1, 1, 1, 1, 1))


@pytest.fixture()
def testvz6000_2(testlasvz6000_2):
    return PointCloud.from_file(testlasvz6000_2, timestamp=datetime.datetime(2022, 1, 1, 2, 2, 2))


@pytest.fixture()
def testdataset_vz6000(testvz6000_1, testvz6000_2) -> Dataset:
    return Dataset.from_instance("pointclouds", [testvz6000_1, testvz6000_2])


@pytest.fixture
def test_sets(request):
    """for testing with different datasets"""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[ROS1FILE, ROS2FILE, ROS2MCAPFILE, ROS2MCAPFILE_LIVOX])
def ros_files(request):
    return request.param
