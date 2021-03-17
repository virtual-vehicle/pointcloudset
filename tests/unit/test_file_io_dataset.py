import pytest_check as check

from lidar import Dataset


def test_from_bag(testbag1):
    ds = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=False)
    check.is_instance(ds, Dataset)


def test_from_bag2(testbag1):
    ds = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    check.is_instance(ds, Dataset)