import pytest_check as check

from lidar import Dataset, Frame
from pathlib import Path
import pandas as pd
import dask.delayed as dd


def test_from_bag(testbag1):
    ds = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=False)
    check.is_instance(ds, Dataset)


def test_from_bag2(testbag1):
    ds = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    check.is_instance(ds, Dataset)


def test_to_dir(testbag1, tmp_path: Path):
    ds = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    testfile_name = tmp_path.joinpath("dataset")
    ds.to_file(file_path=testfile_name)
    check.equal(testfile_name.exists(), True)
    read_dataset = Dataset.from_file(testfile_name)
    check.is_instance(read_dataset, Dataset)
    check.equal(len(ds), len(read_dataset))
