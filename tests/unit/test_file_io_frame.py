from pathlib import Path

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
