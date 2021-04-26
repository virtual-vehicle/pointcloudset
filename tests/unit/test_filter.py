import numpy as np
import pytest
import pytest_check as check

from pointcloudset import PointCloud


def test_wrong_filter(testframe_mini_real: PointCloud):
    with pytest.raises(ValueError):
        testframe_mini_real.filter("not available")


def test_qf1(testframe_mini_real: PointCloud):
    test = testframe_mini_real.filter("quantile", "range", ">", 0.5)
    check.equal(len(test), 12)
    check.greater(test.data.range.min(), 1036.0)


def test_qf2(testframe_mini_real):
    test = testframe_mini_real.filter("quantile", "range", "==", 0.5)
    check.equal(len(test), 0)


def test_qf3(testframe_mini_real: PointCloud):
    q = testframe_mini_real.data.quantile(0.5)
    test = testframe_mini_real.filter("quantile", "range", "<", 0.5)
    check.less(test.data.range.min(), q.range)


def test_qf4(testframe_mini_real: PointCloud):
    q = testframe_mini_real.data.quantile(0.5)
    test = testframe_mini_real.filter("quantile", "range", ">=", 0.5)
    check.greater_equal(test.data.range.min(), q.range)


def test_qf5(testframe_mini_real: PointCloud):
    q = testframe_mini_real.data.quantile(0.5)
    test = testframe_mini_real.filter("quantile", "range", "<=", 0.5)
    check.less_equal(test.data.range.min(), q.range)
