from pathlib import Path

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
import pytest
import pytest_check as check

from pointcloudset import Dataset, PointCloud
from pointcloudset.io.dataset import dir


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


def test_dataset_with_empty_frame(testpointcloud_mini_real: PointCloud, tmp_path: Path):
    fake_empty_df = pd.DataFrame.from_dict(
        {
            "x": [np.nan],
            "y": [np.nan],
            "z": [np.nan],
            "intensity": [np.nan],
            "t": [np.nan],
            "reflectivity": [np.nan],
            "ring": [np.nan],
            "noise": [np.nan],
            "range": [np.nan],
            "original_id": [np.nan],
        }
    )
    pc_empty = PointCloud(data=fake_empty_df)
    testfile_name = tmp_path.joinpath("dataset")
    ds = Dataset.from_instance("pointclouds", [testpointcloud_mini_real, pc_empty])
    ds.to_file(file_path=testfile_name, use_orig_filename=False)
    check.equal(testfile_name.exists(), True)
    read_dataset = Dataset.from_file(testfile_name)
    check.is_instance(read_dataset, Dataset)
    check.equal(len(ds), len(read_dataset))


def test_testdataset_with_empty_frame_r_and_w(
    testdataset_with_empty_frame: Dataset, tmp_path: Path
):
    testfile_name = tmp_path.joinpath("dataset0")

    testdataset_with_empty_frame.to_file(
        file_path=testfile_name, use_orig_filename=False
    )

    data_0_orig = testdataset_with_empty_frame[0].data
    check.equal(testfile_name.exists(), True)
    check.equal(
        len(list(testfile_name.glob("*.parquet"))), len(testdataset_with_empty_frame)
    )
    read_dataset = Dataset.from_file(testfile_name)
    data_0_read = read_dataset[0].data
    check.is_instance(read_dataset, Dataset)
    check.equal(len(testdataset_with_empty_frame), len(read_dataset))
    check.is_false(testdataset_with_empty_frame[1]._has_data())
    assert_frame_equal(data_0_orig, data_0_read)


def test_check_dir_file():
    with pytest.raises(ValueError):
        dir._check_dir(Path("fake.txt"))


def test_check_meta_file(testset: Dataset, tmp_path: Path):
    testfile_name = tmp_path.joinpath("dataset0")
    testset.to_file(testfile_name, use_orig_filename=False)
    testfile_name.joinpath("meta.json").unlink()
    with pytest.raises(AssertionError):
        dir._check_dir_contents_single(testfile_name)
