import pytest
import pytest_check as check

from lidar import Dataset, Frame
import datetime


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
    check.is_instance(test, Dataset)
    check.equal(len(test), 2)
    check.equal(type(testset[0:2][0]), Frame)


def test_getitem_error(testset: Dataset):
    with pytest.raises(TypeError):
        testset["fake"]


def test_getitem_timerange(testset):
    dataset = testset.get_frames_between_timestamps(
        1592833242.6053855, 1592833242.7881582
    )
    check.equal(type(dataset[0:2]), list)
    check.equal(type(dataset[0:2][0]), Frame)


def test_getitem_error3(testset):
    with pytest.raises(ValueError):
        testset[2:0]


def test_has_frames(testset: Dataset):
    check.equal(testset.has_frames(), True)


def test_str(testset: Dataset):
    check.equal(type(str(testset)), str)
    check.equal(
        str(testset),
        "Lidar Dataset with 2 frame(s), from file /workspaces/lidar/tests/testdata/test.bag",
    )


def test_repr(testset: Dataset):
    check.equal(type(str(testset)), str)
    check.equal(repr(testset), "Dataset(/workspaces/lidar/tests/testdata/test.bag)")


def test_testset_len_withzero(testset_withzero):
    check.equal(testset_withzero.keep_zeros, True)
    check.equal(len(testset_withzero), 2)


def test_start_time(testset: Dataset):
    check.is_instance(testset.start_time, datetime.datetime)


def test_first_frame_time(testset: Dataset):
    frame0 = testset[0]
    check.equal(testset._first_frame_time, frame0.timestamp.to_sec())


def test_end_time(testset: Dataset):
    check.equal(testset.end_time, 1592833242.7881582)
    check.greater(testset.end_time, testset.start_time)


def test_time(testset: Dataset):
    check.greater(testset.end_time, testset.start_time)
