import numpy as np
import pytest
import pytest_check as check

from pointcloudset import PointCloud


def test_pointcloud_cluster(testframe_mini_real: PointCloud):
    label = testframe_mini_real.get_cluster(eps=0.8, min_points=5)
    label_ref = np.array(
        [
            [0],
            [0],
            [0],
            [0],
            [0],
            [-1],
            [-1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [-1],
            [-1],
        ]
    )
    np.testing.assert_equal(label.to_numpy(), label_ref)


def test_pointcloud_take_cluster(testframe_mini_real: PointCloud):
    label = testframe_mini_real.get_cluster(eps=0.8, min_points=5)
    cluster0 = testframe_mini_real.take_cluster(0, label)
    check.equal(len(cluster0), 5)
    check.equal(type(cluster0), PointCloud)
    check.equal(testframe_mini_real.timestamp, cluster0.timestamp)
    check.equal(cluster0.data.index.is_monotonic_increasing, True)
