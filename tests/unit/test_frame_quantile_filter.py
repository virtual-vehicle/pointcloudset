import numpy as np
import pytest
import pytest_check as check

from lidar import Frame


def test_qf1(testframe_mini_real):
    test = testframe_mini_real.quantile_filter("range", ">", 0.5)
    check.equal(len(test), 12)
    check.greater(test.data.range.min(), 1036.0)


# TODO: more here for each operator
