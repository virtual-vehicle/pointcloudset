import pytest_check as check

from lidar import Dataset, Frame


def test_init(testbag1):
    dataset = Dataset(testbag1, "ouster")
    check.equal(type(dataset), Dataset)


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


def test_has_frames(testset: Dataset):
    check.equal(testset.has_frames(), True)


def test_str(testset: Dataset):
    check.equal(type(str(testset)), str)
