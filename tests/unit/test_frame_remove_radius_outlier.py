import numpy as np
import pytest
import pytest_check as check

from pointcloudset import Frame


def test_rro1(testframe_mini_real):
    test = testframe_mini_real.filter("radiusoutlier", nb_points=500, radius=0.1)
    check.equal(test._has_data(), False)


def test_rro2(testframe_mini_real):
    test = testframe_mini_real.filter("radiusoutlier", nb_points=10, radius=0.2)
    check.equal(test._has_data(), True)
    check.equal(len(test), 4)
    test_x = test.data["x"].to_list()
    truth = [
        -0.8706228137016296,
        -0.8705275058746338,
        -0.8591182827949524,
        -0.8678007125854492,
    ]
    zipped = zip(test_x, truth)
    test = list(zipped)
    [check.almost_equal(a, b) for a, b in zipped]
