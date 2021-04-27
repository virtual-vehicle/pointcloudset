from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_check as check

import pointcloudset
from pointcloudset import PointCloud


def test_from_file_not_path():
    with pytest.raises(TypeError):
        pointcloudset.PointCloud.from_file("/sepp.depp")


def test_from_file_not_supported(testlas1: Path):
    with pytest.raises(ValueError):
        pointcloudset.PointCloud.from_file(Path("/sepp.depp"))


def test_from_file_las(testlas1: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlas1)
    check.equal(type(pointcloud), PointCloud)
    testdata = Path.cwd().joinpath("tests/testdata/diamond.las").as_posix()
    check.equal(pointcloud.orig_file, testdata)
    check.less_equal(pointcloud.timestamp, datetime.now())
    check.equal(len(list(pointcloud.data.columns)), 13)


def test_to_csv(testpointcloud: PointCloud, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test.csv")
    testpointcloud.to_file(file_path=testfile_name)
    check.equal(testfile_name.exists(), True)
    read_pointcloud = pd.read_csv(testfile_name)
    test_values = read_pointcloud.iloc[0].values
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
            4.624000e03,
        ],
        test_values,
        rtol=1e-10,
        atol=0,
    )


def test_to_csv2(testpointcloud: PointCloud, tmp_path: Path):
    testpointcloud.orig_file = tmp_path.joinpath("fake.bag").as_posix()
    testfile_name = tmp_path.joinpath("fake_timestamp_1592833242755911566.csv")
    testpointcloud.to_file(file_path=testfile_name)
    check.equal(testfile_name.exists(), True)
    read_pointcloud = pd.read_csv(testfile_name)
    test_values = read_pointcloud.iloc[0].values
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
            4.624000e03,
        ],
        test_values,
        rtol=1e-10,
        atol=0,
    )
