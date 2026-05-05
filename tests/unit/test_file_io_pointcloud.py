from datetime import datetime, UTC
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
    check.less_equal(pointcloud.timestamp, datetime.now(UTC))
    check.equal(len(list(pointcloud.data.columns)), 13)


def test_from_file_las_timestamp_file(testlas1: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlas1, timestamp="from_file")
    check.equal(type(pointcloud), PointCloud)
    file_timestamp = datetime.fromtimestamp(testlas1.stat().st_mtime, UTC)
    check.equal(pointcloud.timestamp, file_timestamp)


def test_from_file_las_timestamp_default(testlas1: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testlas1)
    check.equal(type(pointcloud), PointCloud)
    file_timestamp = datetime.fromtimestamp(testlas1.stat().st_mtime, UTC)
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


def test_from_file_xyz(testxyz_diamond: Path):
    with pytest.warns(UserWarning, match="Assuming first three columns are x, y, z"):
        pointcloud = pointcloudset.PointCloud.from_file(testxyz_diamond)
    check.equal(type(pointcloud), PointCloud)
    check.greater(len(pointcloud), 0)
    check.is_true({"x", "y", "z"}.issubset(set(pointcloud.data.columns)))
    np.testing.assert_allclose(pointcloud.data.loc[0, ["x", "y", "z"]].to_numpy(), [0.5, 0.0, 0.5])


def test_from_file_xyz_with_header(testxyz_with_header: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testxyz_with_header)
    check.equal(type(pointcloud), PointCloud)
    check.equal(list(pointcloud.data.columns), ["x", "y", "z", "intensity"])
    np.testing.assert_allclose(pointcloud.data[["x", "y", "z"]].to_numpy(), [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])


def test_from_file_csv_diamond(testcsv_diamond: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testcsv_diamond)
    check.equal(type(pointcloud), PointCloud)
    check.greater(len(pointcloud), 0)
    check.is_true({"x", "y", "z"}.issubset(set(pointcloud.data.columns)))


def test_from_file_csv_without_header(testcsv_headerless: Path):
    with pytest.warns(UserWarning, match="Assuming first three columns are x, y, z"):
        pointcloud = pointcloudset.PointCloud.from_file(testcsv_headerless)

    check.equal(type(pointcloud), PointCloud)
    check.equal(list(pointcloud.data.columns), ["x", "y", "z", "field_3"])
    np.testing.assert_allclose(
        pointcloud.data[["x", "y", "z", "field_3"]].to_numpy(), [[0.5, 0.0, 0.5, 1.0], [0.0, 0.5, 0.5, 2.0]]
    )


def test_from_file_csv_with_header(testcsv_with_header: Path):
    pointcloud = pointcloudset.PointCloud.from_file(testcsv_with_header)
    check.equal(type(pointcloud), PointCloud)
    check.equal(list(pointcloud.data.columns), ["x", "y", "z", "intensity"])
    np.testing.assert_allclose(pointcloud.data[["x", "y", "z"]].to_numpy(), [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])


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


def test_csv_read_write_read_without_header(testpointcloud_mini: PointCloud, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test_without_header.csv")
    testpointcloud_mini.to_file(file_path=testfile_name, header=False)
    check.equal(testfile_name.exists(), True)
    check.is_false(testfile_name.read_text().splitlines()[0].startswith("x,"))

    with pytest.warns(UserWarning, match="Assuming first three columns are x, y, z"):
        read_pointcloud = pointcloudset.PointCloud.from_file(testfile_name)
    check.equal(type(read_pointcloud), PointCloud)
    check.equal(len(read_pointcloud), len(testpointcloud_mini))

    expected = testpointcloud_mini.data.to_numpy()
    actual = read_pointcloud.data.to_numpy()
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-9)


def test_csv_read_write_read_with_header(testpointcloud_mini: PointCloud, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test_with_header.csv")
    testpointcloud_mini.to_file(file_path=testfile_name, header=True)
    check.equal(testfile_name.exists(), True)
    check.is_true(testfile_name.read_text().splitlines()[0].startswith("x,"))

    read_pointcloud = pointcloudset.PointCloud.from_file(testfile_name)
    check.equal(type(read_pointcloud), PointCloud)
    check.equal(len(read_pointcloud), len(testpointcloud_mini))

    expected = testpointcloud_mini.data.to_numpy()
    actual = read_pointcloud.data.to_numpy()
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-9)


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


def test_xyz_read_write_read_target(testpointcloud_mini: PointCloud, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test.xyz")
    testpointcloud_mini.to_file(file_path=testfile_name)
    check.equal(testfile_name.exists(), True)
    check.is_false(testfile_name.read_text().splitlines()[0].startswith("x "))

    with pytest.warns(UserWarning, match="Assuming first three columns are x, y, z"):
        read_pointcloud = pointcloudset.PointCloud.from_file(testfile_name)
    check.equal(type(read_pointcloud), PointCloud)
    check.equal(len(read_pointcloud), len(testpointcloud_mini))

    expected = testpointcloud_mini.data[["x", "y", "z"]].to_numpy()
    actual = read_pointcloud.data[["x", "y", "z"]].to_numpy()
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-9)


def test_xyz_read_write_read_with_header(testpointcloud_mini: PointCloud, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test_with_header.xyz")
    testpointcloud_mini.to_file(file_path=testfile_name, header=True)
    check.equal(testfile_name.exists(), True)
    check.is_true(testfile_name.read_text().splitlines()[0].startswith("x y z"))

    read_pointcloud = pointcloudset.PointCloud.from_file(testfile_name)
    check.equal(type(read_pointcloud), PointCloud)
    check.equal(len(read_pointcloud), len(testpointcloud_mini))

    expected = testpointcloud_mini.data[["x", "y", "z"]].to_numpy()
    actual = read_pointcloud.data[["x", "y", "z"]].to_numpy()
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-9)


def test_pcd_read_write_read_target(testpointcloud_mini: PointCloud, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test.pcd")
    testpointcloud_mini.to_file(file_path=testfile_name)
    check.equal(testfile_name.exists(), True)

    read_pointcloud = pointcloudset.PointCloud.from_file(testfile_name)
    check.equal(type(read_pointcloud), PointCloud)
    check.equal(len(read_pointcloud), len(testpointcloud_mini))

    expected = testpointcloud_mini.data[["x", "y", "z"]].to_numpy()
    actual = read_pointcloud.data[["x", "y", "z"]].to_numpy()
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-9)


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
