from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_check as check

import lidar
from lidar import Frame


def test_from_file_not_path():
    with pytest.raises(TypeError):
        lidar.Frame.from_file("/sepp.depp")


def test_from_file_not_supported(testlas1: Path):
    with pytest.raises(ValueError):
        lidar.Frame.from_file(Path("/sepp.depp"))


def test_from_file_las(testlas1: Path):
    frame = lidar.Frame.from_file(testlas1)
    check.equal(type(frame), Frame)
    check.equal(frame.orig_file, "/workspaces/lidar/tests/testdata/diamond.las")
    check.equal(frame.timestamp_str, "Tuesday, December 01, 2020 11:32:41")
    check.equal(
        list(frame.data.columns),
        [
            "x",
            "y",
            "z",
            "intensity",
            "flag_byte",
            "raw_classification",
            "scan_angle_rank",
            "user_data",
            "pt_src_id",
            "gps_time",
            "red",
            "green",
            "blue",
        ],
    )


def test_to_csv(testframe: Frame, tmp_path: Path):
    testfile_name = tmp_path.joinpath("just_test.csv")
    testframe.to_file(file_path=testfile_name)
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
            4.624000e03,
        ],
        test_values,
        rtol=1e-10,
        atol=0,
    )


def test_to_csv2(testframe: Frame, tmp_path: Path):
    testframe.orig_file = tmp_path.joinpath("fake.bag").as_posix()
    testfile_name = tmp_path.joinpath("fake_timestamp_1592833242755911566.csv")
    testframe.to_file(file_path=testfile_name)
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
            4.624000e03,
        ],
        test_values,
        rtol=1e-10,
        atol=0,
    )
