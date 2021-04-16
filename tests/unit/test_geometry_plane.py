import numpy as np
import pytest
import pytest_check as check
from numpy.testing import assert_array_equal

from lidar.geometry import plane


def test_distance_to_point1():
    distance = plane.distance_to_point(
        point_A=np.array([1, 0, 0]), plane_model=np.array([1, 0, 0, 0])
    )
    check.equal(distance, 1.0)


def test_distance_to_point2():
    distance = plane.distance_to_point(
        point_A=np.array([2, 2, 0]),
        plane_model=np.array([1, 0, 0, -1]),
        normal_dist=False,
    )
    check.equal(distance, (np.sqrt(2)))


def test_distance_to_point3():
    distance = plane.distance_to_point(
        point_A=np.array([2, 2, 0]),
        plane_model=np.array([1, 0, 0, -1]),
        normal_dist=True,
    )
    check.equal(distance, 1.0)


def test_distance_to_point_error1():
    with pytest.raises(ValueError):
        plane.distance_to_point(
            point_A=np.array([1, 0]), plane_model=np.array([1, 0, 0, 0])
        )


def test_distance_to_point_error2():
    with pytest.raises(ValueError):
        plane.distance_to_point(
            point_A=np.array([1, 0, 0]),
            plane_model=np.array(
                [
                    1,
                    0,
                    0,
                ]
            ),
        )


def test_intersect_line_of_sight():
    point = plane.intersect_line_of_sight(
        line=np.array([1, 1, 1]), plane_model=np.array([0, 0, 1, -1])
    )
    np.testing.assert_allclose(point, np.array([1.0, 1.0, 1.0]))


def test_intersect_line_of_sight_error1():
    with pytest.raises(ValueError):
        plane.intersect_line_of_sight(
            line=np.array([1, 0]), plane_model=np.array([1, 0, 0, 0])
        )


def test_intersect_line_of_sight_error2():
    with pytest.raises(ValueError):
        plane.intersect_line_of_sight(
            line=np.array([1, 0, 0]),
            plane_model=np.array(
                [
                    1,
                    0,
                    0,
                ]
            ),
        )