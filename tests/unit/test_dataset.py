import pytest
import pytest_check as check

from lidar import Dataset, Frame


def test_init(testbag1):
    dataset = Dataset(testbag1, "/os1_cloud_node/points")
    check.equal(type(dataset), Dataset)


def test_init_error(testbag1):
    with pytest.raises(IOError):
        dataset = Dataset(testbag1, "/fake")


def test_testset(testset):
    check.equal(len(testset), 2)
    check.equal(type(testset.orig_file), str)


def test_testset_topic(testset):
    topic = testset.topic
    check.equal(type(topic), str)
    check.equal(topic, "/os1_cloud_node/points")


def test_testset_keep_zeros(testset):
    check.equal(testset.keep_zeros, False)


def test_testset_keep_zeros_true(testset_withzero):
    check.equal(testset_withzero.keep_zeros, True)


def test_getitem(testset: Dataset):
    check.equal(type(testset[0]), Frame)


def test_getitem_late(testset):
    check.equal(type(testset[1]), Frame)


def test_getitem_slice(testset):
    check.equal(type(testset[0:2]), list)
    check.equal(type(testset[0:2][0]), Frame)


def test_getitem_error(testset: Dataset):
    with pytest.raises(TypeError):
        testset["fake"]


def test_getitem_timerange(testbag1):
    dataset = Dataset(
        testbag1,
        topic="/os1_cloud_node/points",
        timerange=(1592833242.6053855, 1592833242.7881582),
    )
    check.equal(type(dataset.timerange), tuple)
    check.equal(type(dataset[0:2]), list)
    check.equal(type(dataset[0:2][0]), Frame)


def test_getitem_error3(testset):
    with pytest.raises(ValueError):
        testset[2:0]


def test_has_frames(testset: Dataset):
    check.equal(testset.has_frames(), True)


def test_str(testset: Dataset):
    check.equal(type(str(testset)), str)


def test_size(testset: Dataset):
    check.equal(testset.size, 16040180)


def test_start_time(testset: Dataset):
    check.equal(testset.start_time, 1592833242.6053855)


def test_end_time(testset: Dataset):
    check.equal(testset.end_time, 1592833242.7881582)


def test_time(testset: Dataset):
    check.greater(testset.end_time, testset.start_time)
