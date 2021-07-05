from pathlib import Path

import pytest
import pytest_check as check

from pointcloudset import Dataset


def test_from_bag_wrong_topic(testbag1):
    with pytest.raises(KeyError):
        ds = Dataset.from_file(testbag1, topic="/none", keep_zeros=False)


def test_from_bag(testbag1):
    ds = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=False)
    check.is_instance(ds, Dataset)


def test_from_bag2(testbag1):
    ds = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    check.is_instance(ds, Dataset)


def test_to_dir(testbag1, tmp_path: Path):
    ds = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    testfile_name = tmp_path.joinpath("dataset")
    ds.to_file(file_path=testfile_name, use_orig_filename=False)
    check.equal(testfile_name.exists(), True)
    read_dataset = Dataset.from_file(testfile_name)
    check.is_instance(read_dataset, Dataset)
    check.equal(len(ds), len(read_dataset))


def test_empty_dataset(tmp_path: Path):
    complete_empty_dataset = Dataset()
    check.is_false(complete_empty_dataset.has_pointclouds())
    testfile_name = tmp_path.joinpath("dataset0")
    with pytest.raises(ValueError):
        complete_empty_dataset.to_file(file_path=testfile_name, use_orig_filename=False)
