import numpy as np
import pytest
import pytest_check as check

from lidar import Frame


def test_rro1(testframe_mini_real):
    test = testframe_mini_real.remove_radius_outlier(nb_points=500, radius=0.1)
    check.equal(test.has_data(), False)


def test_rro2(testframe_mini_real):
    test = testframe_mini_real.remove_radius_outlier(nb_points=5, radius=0.1)
    check.equal(test.has_data(), True)
    check.equal(len(test), 3)
