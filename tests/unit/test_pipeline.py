import pytest
import pytest_check as check

from lidar import Dataset, Frame


def test_pipeline1(testset: Dataset):
    def pipelin1(frame: Frame):
        return frame.limit("x", 0, 1)

    testset_result = testset.apply_pipeline(pipeline=pipelin1)
    check.equal(type(testset_result), list)
    check.equal(len(testset_result), len(testset))
    check.less_equal(max(testset_result[0].data["x"]), 1.0)
    check.less_equal(max(testset_result[1].data["x"]), 1.0)
