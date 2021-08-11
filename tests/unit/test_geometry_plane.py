import numpy as np
import pytest
import pytest_check as check
from numpy.testing import assert_array_equal

from pointcloudset.geometry import plane


@pytest.mark.parametrize(
    "point, res", [([0, 0, 0], 10), ([1, 1, 1], 11), ([-1, -1, -1], 9)]
)
def test_distance_to_point(point, res):
    distance = plane.distance_to_point(
        point_A=np.array(point), plane_model=np.array([1, 0, 0, 10])
    )
    check.equal(distance, res)


@pytest.mark.parametrize("normal_dist, res", [(False, np.sqrt(2)), (True, 1.0)])
def test_distance_to_point_normal_dist(normal_dist, res):
    distance = plane.distance_to_point(
        point_A=np.array([2, 2, 0]),
        plane_model=np.array([1, 0, 0, -1]),
        normal_dist=normal_dist,
    )
    check.equal(distance, res)


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
