from pathlib import Path

import pytest_check as check

import lidar
from lidar import Frame


def test_from_file_las(testlas1: Path):
    frame = lidar.Frame.from_file(testlas1)
    check.equal(type(frame), Frame)
    check.equal(frame.orig_file, "/workspaces/lidar/tests/testdata/diamond.las")
