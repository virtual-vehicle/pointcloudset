import numpy as np
import pytest
import pytest_check as check

from lidar.geometry import point


def test_distance_to_point():
    distance = point.distance_to_point(
        point_A=np.array([1, 0, 0]), point_B=np.array([2, 0, 0])
    )
    check.equal(distance, 1.0)


def test_distance_to_point_error1():
    with pytest.raises(ValueError):
        point.distance_to_point(point_A=np.array([1, 0]), point_B=np.array([1, 0, 0]))


def test_distance_to_point_error2():
    with pytest.raises(ValueError):
        point.distance_to_point(point_A=np.array([1, 0, 0]), point_B=np.array([1, 0]))