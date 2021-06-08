import numpy as np
import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import PointCloud
from pointcloudset.diff.pointcloud import _calculate_single_point_difference


def test_pointcloud_difference_equal(testpointcloud_mini_real: PointCloud):
    res = testpointcloud_mini_real.diff("pointcloud", testpointcloud_mini_real)
    x_test = res.data["x difference"].values
    y_test = res.data["y difference"].values
    z_test = res.data["z difference"].values
    check.equal(type(res), PointCloud)
    check.equal(len(res), len(testpointcloud_mini_real))
    check.equal(
        list(res.data.columns),
        [
            "x",
            "y",
            "z",
            "intensity",
            "t",
            "reflectivity",
            "ring",
            "noise",
            "range",
            "original_id",
            "x difference",
            "y difference",
            "z difference",
            "intensity difference",
            "t difference",
            "reflectivity difference",
            "ring difference",
            "noise difference",
            "range difference",
        ],
    )
    check.equal(np.all(x_test == 0.0), True)
    check.equal(np.all(y_test == 0.0), True)
    check.equal(np.all(z_test == 0.0), True)


def test_pointcloud_difference1_no_orignal_id(
    testpointcloud_mini: PointCloud, testpointcloud_mini_real: PointCloud
):
    with pytest.raises(ValueError):
        testpointcloud_mini.diff("pointcloud", testpointcloud_mini_real)


def test_pointcloud_difference1_no_orignal_id2(
    testpointcloud_mini: PointCloud, testpointcloud_mini_real: PointCloud
):
    with pytest.raises(ValueError):
        testpointcloud_mini_real.diff("pointcloud", testpointcloud_mini)


def test_pointcloud_difference1(
    testpointcloud_mini_real: PointCloud, testpointcloud_mini_real_plus1: PointCloud
):
    res = testpointcloud_mini_real.diff("pointcloud", testpointcloud_mini_real_plus1)
    check.equal(type(res), PointCloud)
    check.equal(len(res), len(testpointcloud_mini_real))
    x_test = res.data["x difference"].values
    y_test = res.data["y difference"].values
    z_test = res.data["z difference"].values
    check.equal(np.all(np.isclose(x_test, -1.0)), True)
    check.equal(np.all(np.isclose(y_test, -1.0)), True)
    check.equal(np.all(np.isclose(z_test, -1.0)), True)


def test_pointcloud_difference_no_intersection(
    testpointcloud_mini_real: PointCloud,
    testpointcloud_mini_real_other_original_id: PointCloud,
):
    with pytest.raises(ValueError):
        testpointcloud_mini_real.diff(
            "pointcloud", testpointcloud_mini_real_other_original_id
        )


def test__calculate_single_point_difference_no_overlap(
    testpointcloud_mini_real: PointCloud,
    testpointcloud_mini_real_other_original_id: PointCloud,
):
    res = _calculate_single_point_difference(
        testpointcloud_mini_real,
        testpointcloud_mini_real_other_original_id,
        original_id=6008,
    )
    test = res.values[0]
    check.is_instance(res, pd.DataFrame)
    check.equal(np.alltrue(np.isnan(test)), True)


def test_distances_to_origin(testpointcloud_mini: PointCloud):
    newframe = testpointcloud_mini.diff("origin")
    check.equal(type(newframe), PointCloud)
    check.equal(
        np.allclose(
            testpointcloud_mini.data["distance to point: [0 0 0]"].values,
            np.asarray(
                [
                    0.0,
                    1.73205081,
                    1.73205081,
                    937.15579489,
                    1058.09727232,
                    906.10064672,
                    991.83926827,
                    475.99837556,
                ]
            ),
        ),
        True,
    )


def test_calculate_distance_to_plane1(testpointcloud_mini: PointCloud):
    newframe = testpointcloud_mini.diff(
        "plane", target=np.array([1, 0, 0, 0]), absolute_values=False
    )
    check.equal(type(newframe), PointCloud)
    check.equal(
        str(list(testpointcloud_mini.data.columns.values)),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'distance to plane: [1 0 0 0]']",
    )
    check.equal(testpointcloud_mini.data["distance to plane: [1 0 0 0]"][1], 1.0)


def test_calculate_distance_to_plane1_2(testpointcloud_mini: PointCloud):
    newframe = testpointcloud_mini.diff("plane", np.array([1, 0, 0, 0]))
    check.equal(type(newframe), PointCloud)
    check.equal(
        str(list(testpointcloud_mini.data.columns.values)),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'distance to plane: [1 0 0 0]']",
    )
    check.equal(testpointcloud_mini.data["distance to plane: [1 0 0 0]"][1], 1.0)


def test_calculate_distance_to_plane2(testpointcloud_mini: PointCloud):
    testpointcloud_mini.diff(
        "plane", target=np.array([-1, 0, 0, 0]), absolute_values=False
    )
    check.equal(
        str(list(testpointcloud_mini.data.columns.values)),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'distance to plane: [-1 0 0 0]']",
    )
    check.equal(testpointcloud_mini.data["distance to plane: [-1 0 0 0]"][1], -1.0)


def test_calculate_distance_to_plane3(testpointcloud_mini: PointCloud):
    testpointcloud_mini.diff(
        "plane", target=np.array([-1, 0, 0, 0]), absolute_values=True
    )
    check.equal(testpointcloud_mini.data["distance to plane: [-1 0 0 0]"][1], 1.0)


def test_calculate_distance_to_point(testpointcloud_mini: PointCloud):
    testpointcloud_mini.diff("point", target=np.array([-1, 0, 0]))
    check.equal(testpointcloud_mini.data["distance to point: [-1 0 0]"][0], 1.0)


def test_pointcloud_diff_of_diff(testpointcloud_mini_real: PointCloud):
    res = testpointcloud_mini_real.diff("pointcloud", testpointcloud_mini_real)
    with pytest.raises(NotImplementedError):
        res.diff("pointcloud", testpointcloud_mini_real)
