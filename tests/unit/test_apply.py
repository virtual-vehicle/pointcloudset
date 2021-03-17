import pytest_check as check

from lidar import Dataset, Frame


def test_apply1(testset: Dataset):
    def pipeline1(frame: Frame) -> Frame:
        return frame.limit("x", 0, 1)

    testset_result = testset.apply(func=pipeline1)
    check.equal(type(testset_result), Dataset)
    check.equal(len(testset_result), len(testset))
    check.less_equal(max(testset_result[0].data["x"]), 1.0)
    check.less_equal(max(testset_result[1].data["x"]), 1.0)


def test_apply2(testset: Dataset):
    def pipeline1(frame: Frame):
        return frame.data.x.max()

    testset_result = testset.apply(
        func=pipeline1,
    )
    check.equal(type(testset_result), tuple)
    check.equal(len(testset_result), len(testset))
    check.almost_equal(testset_result, (13.721149, 13.744355))


def test_apply_both(testset: Dataset):
    def pipeline1(frame: Frame) -> Frame:
        return frame.limit("x", 0, 1)

    def pipeline2(frame: Frame):
        return frame.data.x.max()

    testset_result = testset.apply(pipeline1).apply(pipeline2)
    check.equal(type(testset_result), tuple)
    check.equal(len(testset_result), len(testset))
    check.less(max(testset_result), 1)
    check.greater_equal(max(testset_result), 0)


def test_apply3(testset: Dataset):
    def pipeline1(frame: Frame) -> Frame:
        return frame.limit("x", 0, 1)

    testset2 = testset[0:2]
    testset_result = testset2.apply(func=pipeline1)
    check.equal(type(testset_result), Dataset)
    check.equal(len(testset_result), 2)
