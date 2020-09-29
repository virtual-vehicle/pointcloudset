import pytest_check as check

from lidar import Dataset, Frame
from lidar.processing.pipeline import apply_pipeline


def test_pipeline1(testset: Dataset):
    test_list = [testset[0], testset[1]]

    def pipeline1(frame: Frame):
        return frame.limit("x", 0, 1)

    testset_result = apply_pipeline(test_list, pipeline=pipeline1)
    check.equal(type(testset_result), list)
    check.equal(len(testset_result), len(test_list))
    check.less_equal(max(testset_result[0].data["x"]), 1.0)
    check.less_equal(max(testset_result[1].data["x"]), 1.0)
