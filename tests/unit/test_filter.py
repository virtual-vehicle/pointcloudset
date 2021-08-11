import numpy as np
import pytest
import pytest_check as check

from pointcloudset import PointCloud
from pointcloudset.config import OPS


def test_wrong_filter(testpointcloud_mini_real: PointCloud):
    with pytest.raises(ValueError):
        testpointcloud_mini_real.filter("not available")


def test_qf1(testpointcloud_mini_real: PointCloud):
    test = testpointcloud_mini_real.filter("quantile", "range", ">", 0.5)
    check.equal(len(test), 12)
    check.greater(test.data.range.min(), 1036.0)


def test_qf2(testpointcloud_mini_real):
    test = testpointcloud_mini_real.filter("quantile", "range", "==", 0.5)
    check.equal(len(test), 0)


@pytest.mark.parametrize("op", ["<", ">=", "<="])
def test_qf3(testpointcloud_mini_real: PointCloud, op):
    q = testpointcloud_mini_real.data.quantile(0.5)
    test = testpointcloud_mini_real.filter("quantile", "range", op, 0.5)
    assert OPS[op](test.data.range.min(), q.range)
