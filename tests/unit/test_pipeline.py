import pytest_check as check

from lidar import Dataset, Frame


def test_pipeline1(testset: Dataset):
    def pipeline1(frame: Frame, frame_number=0):
        return frame.limit("x", 0, 1)

    testset_result = testset.apply_pipeline(pipeline=pipeline1)
    check.equal(type(testset_result), list)
    check.equal(len(testset_result), len(testset))
    check.less_equal(max(testset_result[0].data["x"]), 1.0)
    check.less_equal(max(testset_result[1].data["x"]), 1.0)


def test_pipeline2(testset: Dataset, frame_number=0):
    def pipeline1(frame: Frame, frame_number=0):
        return frame.limit("x", 0, 1)

    testset_result = testset.apply_pipeline(
        pipeline=pipeline1,
        start_frame_number=0,
    )
    check.equal(type(testset_result), list)
    check.equal(len(testset_result), len(testset))
    check.less_equal(max(testset_result[0].data["x"]), 1.0)
    check.less_equal(max(testset_result[1].data["x"]), 1.0)


def test_pipeline3(testset: Dataset):
    def pipeline1(frame: Frame, frame_number=0):
        return frame.limit("x", 0, 1)

    testset_result = testset.apply_pipeline(
        pipeline=pipeline1, start_frame_number=0, end_frame_number=1
    )
    check.equal(type(testset_result), list)
    check.equal(len(testset_result), 1)
