import datetime

import pytest
import pytest_check as check
import rosbag
from dask.delayed import DelayedLeaf

from lidar import Dataset, Frame


def test_dataset_len(testbag1: str, testset: Dataset):
    bag = rosbag.Bag(testbag1)
    len_bag = (bag.get_type_and_topic_info().topics)[
        "/os1_cloud_node/points"
    ].message_count
    check.equal(len_bag, len(testset))
    check.equal(len(testset), 2)


def test_getitem(testset: Dataset):
    check.equal(type(testset[0]), Frame)


def test_getitem_late(testset):
    check.equal(type(testset[1]), Frame)


def test_getitem_2times(testset):
    check.equal(type(testset[0]), Frame)
    check.equal(type(testset[1]), Frame)
    check.equal(type(testset[0]), Frame)
    check.equal(type(testset[1]), Frame)


def test_getitem_slice(testset: Dataset):
    test = testset[0:2]
    testlist = [0, 1, 2, 3]
    check.is_instance(test, Dataset)
    check.equal(len(test), 2)
    check.equal(len(test), len(testlist[0:2]))
    check.equal(type(testset[0:2][0]), Frame)


def test_getitem_error(testset: Dataset):
    with pytest.raises(TypeError):
        testset["fake"]


def test_get_frame_number_from_time(testset):
    res = testset._get_frame_number_from_time(testset.start_time)
    check.equal(res, 0)
    res2 = testset._get_frame_number_from_time(testset.end_time)
    check.equal(res2 + 1, len(testset))


def test_getitem_timerange(testset: Dataset):
    check.equal(len(testset), 2)
    dataset = testset.get_frames_between_timestamps(
        datetime.datetime(2020, 6, 22, 13, 40, 42, 657267),
        datetime.datetime(2020, 6, 22, 13, 40, 42, 755912),
    )
    check.equal(len(dataset), 2)
    check.equal(type(dataset[0:2]), Dataset)
    check.equal(type(dataset[0:2][0]), Frame)


def test_getitem_strange(testset):
    check.equal(len(testset), 2)
    check.equal(len(testset[2:0]), 0)


def test_has_frames(testset: Dataset):
    check.equal(testset.has_frames(), True)


def test_str(testset: Dataset):
    check.equal(type(str(testset)), str)
    check.equal(
        str(testset),
        "Lidar Dataset with 2 frame(s)",
    )


def test_repr(testset: Dataset):
    repr = testset.__repr__()
    check.equal(type(repr), str)
    check.equal(len(repr), 326)


def test_start_time(testset: Dataset):
    st = testset.start_time
    check.is_instance(st, datetime.datetime)
    check.equal(st, datetime.datetime(2020, 6, 22, 13, 40, 42, 657267))


def test_end_time(testset: Dataset):
    et = testset.end_time
    check.is_instance(et, datetime.datetime)
    check.equal(et, datetime.datetime(2020, 6, 22, 13, 40, 42, 755912))


def test_time(testset: Dataset):
    check.greater(testset.end_time, testset.start_time)


def test_extend(testbag1):
    ds1 = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    len_old = len(ds1)
    ds2 = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    ds1.extend(ds2)
    check.equal(len(ds1), len_old * 2)


def test_from_frames_list(testdataset_mini_real):
    check.is_instance(testdataset_mini_real, Dataset)
    check.equal(len(testdataset_mini_real), 2)
    check.is_instance(testdataset_mini_real.data[0], DelayedLeaf)
    check.is_instance(testdataset_mini_real[0], Frame)
