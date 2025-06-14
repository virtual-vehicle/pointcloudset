from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import pytest
import pytest_check as check

import pointcloudset
from pointcloudset import PointCloud


@pytest.mark.parametrize("file, error", [(Path("/sepp.depp"), "ValueError"), ("/sepp.depp", "TypeError")])
def test_from_file_error(file, error):
    with pytest.raises(eval(error)):
        pointcloudset.PointCloud.from_file(file)


def test_from_file_las(testlas1: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlas1)
    check.equal(type(pointcloud), PointCloud)
    testdata = testlas1.as_posix()
    check.equal(pointcloud.orig_file, testdata)
    check.less_equal(pointcloud.timestamp, datetime.now())
    check.equal(len(list(pointcloud.data.columns)), 13)


def test_from_file_las_timestamp_file(testlas1: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlas1, timestamp="from_file")
    check.equal(type(pointcloud), PointCloud)
    file_timestamp = datetime.utcfromtimestamp(testlas1.stat().st_mtime)
    check.equal(pointcloud.timestamp, file_timestamp)


def test_from_file_las_timestamp_default(testlas1: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlas1)
    check.equal(type(pointcloud), PointCloud)
    file_timestamp = datetime.utcfromtimestamp(testlas1.stat().st_mtime)
    check.equal(pointcloud.timestamp, file_timestamp)


def test_from_file_las_timestamp_insert(testlas1: Path):
    timestamp = datetime(2022, 1, 1, 1, 1)
    pointcloud = pointcloudset.PointCloud.from_file(testlas1, timestamp=timestamp)
    check.equal(type(pointcloud), PointCloud)
    check.equal(pointcloud.timestamp, timestamp)


def test_from_file_las_vz6000_1(testlasvz6000_1: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlasvz6000_1)
    check.equal(type(pointcloud), PointCloud)
    check.is_false(pointcloud.has_original_id)


def test_from_file_las_vz6000_2(testlasvz6000_2: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlasvz6000_2)
    check.equal(type(pointcloud), PointCloud)


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


def test_to_las(testpointcloud: PointCloud, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test.las")
    testpointcloud.to_file(file_path=testfile_name)
    check.equal(testfile_name.exists(), True)
    read_pointcloud = pointcloudset.PointCloud.from_file(testfile_name)
    expected_columns = {
        "x",
        "y",
        "z",
        "intensity",
        "bit_fields",
        "classification_flags",
        "classification",
        "user_data",
        "scan_angle",
        "point_source_id",
        "gps_time",
        "red",
        "green",
        "blue",
        "noise",
        "original_id",
        "range",
        "reflectivity",
        "ring",
        "t",
    }

    actual_columns = set(read_pointcloud.data.columns)
    check.equal(actual_columns, expected_columns)
    test_values = (
        read_pointcloud.data[
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
            ]
        ]
        .iloc[0]
        .values
    )

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
        rtol=1e-4,
        atol=0,
    )


def test_to_laz_not_implemented(testpointcloud: PointCloud, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test.laz")
    with pytest.raises(ValueError):
        testpointcloud.to_file(file_path=testfile_name)


def test_las_read_write_read1(testlas1: Path, tmp_path: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlas1, timestamp="from_file")
    testfile_name = tmp_path.joinpath("just_test.las")
    check.equal(type(pointcloud), PointCloud)
    pointcloud.to_file(file_path=testfile_name)
    check.equal(testfile_name.exists(), True)
    read_pointcloud = pointcloudset.PointCloud.from_file(testfile_name)
    check.equal(type(read_pointcloud), PointCloud)


def test_las_read_write_read_tree(test_las_tree: Path, tmp_path: Path):
    pointcloud = pointcloudset.PointCloud.from_file(test_las_tree, timestamp="from_file")
    testfile_name = tmp_path.joinpath("just_test.las")
    check.equal(type(pointcloud), PointCloud)
    pointcloud.to_file(file_path=testfile_name)
    check.equal(testfile_name.exists(), True)
    read_pointcloud = pointcloudset.PointCloud.from_file(testfile_name)
    check.equal(type(read_pointcloud), PointCloud)
