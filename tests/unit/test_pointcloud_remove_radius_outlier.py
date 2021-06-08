import numpy as np
import pytest
import pytest_check as check

from pointcloudset import PointCloud


def test_rro1(testpointcloud_mini_real):
    test = testpointcloud_mini_real.filter("radiusoutlier", nb_points=500, radius=0.1)
    check.equal(test._has_data(), False)
