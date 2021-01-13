import numpy as np
import pytest
import pytest_check as check

from lidar.geometry import plane


def test_distance_to_point():
    distance = plane.distance_to_point(
        point=np.array([1, 0, 0]), plane_model=np.array([1, 0, 0, 0])
    )
    check.equal(distance, 1.0)


def test_distance_to_point_error1():
    with pytest.raises(ValueError):
        plane.distance_to_point(
            point=np.array([1, 0]), plane_model=np.array([1, 0, 0, 0])
        )


def test_distance_to_point_error2():
    with pytest.raises(ValueError):
        plane.distance_to_point(
            point=np.array([1, 0, 0]),
            plane_model=np.array(
                [
                    1,
                    0,
                    0,
                ]
            ),
        )
