import numpy as np
import pytest
import pytest_check as check

from pointcloudset.geometry import point


def test_distance_to_point():
    distance = point.distance_to_point(
        point_a=np.array([1, 0, 0]), point_b=np.array([2, 0, 0])
    )
    check.equal(distance, 1.0)


@pytest.mark.parametrize("pA, pB", [([1, 0], [1, 0, 0]), ([1, 0, 0], [1, 0])])
def test_distance_to_point_error(pA, pB):
    with pytest.raises(ValueError):
        point.distance_to_point(point_a=np.array(pA), point_b=np.array(pB))
