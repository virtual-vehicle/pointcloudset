import datetime

import pytest_check as check

from pointcloudset import Dataset, PointCloud
from pointcloudset.pipeline.delayed_result import DelayedResult


def test_apply1(testset: Dataset):
    def pipeline1(pointcloud: PointCloud) -> PointCloud:
        return pointcloud.limit("x", 0, 1)

    testset_result = testset.apply(func=pipeline1)
    check.equal(type(testset_result), Dataset)
    check.equal(len(testset_result), len(testset))
    check.less_equal(max(testset_result[0].data["x"]), 1.0)
    check.less_equal(max(testset_result[1].data["x"]), 1.0)


def test_apply2(testset: Dataset):
    def pipeline1(pointcloud: PointCloud):
        return pointcloud.data.x.max()

    testset_result = testset.apply(
        func=pipeline1,
    )
    check.equal(type(testset_result), DelayedResult)
    check.equal(len(testset_result), len(testset))
    check.almost_equal(testset_result.compute(), [13.721149, 13.744355])


def test_apply_both(testset: Dataset):
    def pipeline1(pointcloud: PointCloud) -> PointCloud:
        return pointcloud.limit("x", 0, 1)

    def pipeline2(pointcloud: PointCloud):
        return pointcloud.data.x.max()

    testset_result = testset.apply(pipeline1).apply(pipeline2)
    testset_result2 = testset_result.compute()
    check.equal(type(testset_result), DelayedResult)
    check.equal(len(testset_result2), len(testset))
    check.less(max(testset_result2), 1)
    check.greater_equal(max(testset_result2), 0)


def test_apply3(testset: Dataset):
    def pipeline1(pointcloud: PointCloud) -> PointCloud:
        return pointcloud.limit("x", 0, 1)

    testset2 = testset[0:2]
    testset_result = testset2.apply(func=pipeline1)
    check.equal(type(testset_result), Dataset)
    check.equal(len(testset_result), 2)


def test_apply_time(testset: Dataset):
    def pipeline1(pointcloud: PointCloud):
        return pointcloud.timestamp

    testset2 = testset[0:2]
    testset_result = testset2.apply(func=pipeline1).compute()
    check.equal(testset_result[0], datetime.datetime(2020, 6, 22, 13, 40, 42, 657267))
    check.equal(testset_result[0] < testset_result[1], True)


def test_apply_with_args(testset: Dataset):
    def pipeline1(pointcloud: PointCloud, test):
        return test

    testset2 = testset[0:2]
    testset_result = testset2.apply(func=pipeline1, test=1).compute()
    check.equal(testset_result, [1, 1])


def test_apply_empty_frame_res(testdataset_with_empty_frame: Dataset):
    check.equal(len(testdataset_with_empty_frame), 2)
    first_frame = testdataset_with_empty_frame[0]
    second_frame = testdataset_with_empty_frame[1]
    check.is_true(first_frame._has_data())
    check.is_false(second_frame._has_data())
    check.equal(len(second_frame), 0)
    check.equal(list(first_frame.data.columns), list(second_frame.data.columns))
