import pytest_check as check

from lidar import Dataset, Frame
from typing import List
import rospy


def test_init(testbag1):
    dataset = Dataset(testbag1, "ouster")
    check.equal(type(dataset), Dataset)


def test_testset(testset):
    check.equal(len(testset), 2)
    check.equal(type(testset.orig_file), str)
    check.equal(testset.orig_file, "/fault_injection/lidar/tests/testdata/test.bag")


def test_getitem(testset: Dataset):
    check.equal(type(testset[0]), Frame)


def test_getitem_late(testbag1):
    dataset = Dataset(testbag1, "ouster")
    check.equal(type(dataset[1]), Frame)


def test_has_frames(testset: Dataset):
    check.equal(testset.has_frames(), True)


def test_str(testset: Dataset):
    check.equal(type(str(testset)), str)
    check.equal(
        str(testset),
        "Lidar Dataset with 2 frame(s), from file /fault_injection/lidar/tests/testdata/test.bag",
    )
