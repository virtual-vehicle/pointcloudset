import numpy as np
import pytest
import pytest_check as check

from pointcloudset import PointCloud


def test_pointcloud_limit(testpointcloud_mini: PointCloud):
    check.equal(len(testpointcloud_mini.limit("x", minvalue=0.0, maxvalue=2.0)), 2)
    check.equal(len(testpointcloud_mini.limit("x", minvalue=-100.0, maxvalue=-10)), 0)


def test_pointcloud_limit2(testpointcloud_mini: PointCloud):
    with pytest.raises(KeyError):
        testpointcloud_mini.limit("wrong", minvalue=0.0, maxvalue=200)


def test_pointcloud_limit3(testpointcloud_mini: PointCloud):
    with pytest.raises(ValueError):
        testpointcloud_mini.limit("wrong", minvalue=2000, maxvalue=0)


def test_pointcloud_limit4(testpointcloud_mini: PointCloud, testpointcloud_mini_df):
    np.testing.assert_array_equal(
        testpointcloud_mini.limit("x", minvalue=0.0, maxvalue=2.0).data.values,
        testpointcloud_mini_df[0:2].values,
    )


def test_pointcloud_limit5(testpointcloud_mini: PointCloud, testpointcloud_mini_df):
    np.testing.assert_array_equal(
        testpointcloud_mini.limit("intensity", minvalue=0.0, maxvalue=2.0).data.values,
        testpointcloud_mini_df[0:2].values,
    )


def test_limit_6(testpointcloud: PointCloud):
    totest = testpointcloud.limit("x", minvalue=0.0, maxvalue=500.0)
    check.equal(len(totest), 25778)


def test_limit_7(testpointcloud: PointCloud):
    totest = testpointcloud.limit("intensity", minvalue=500.0, maxvalue=510.0)
    check.equal(len(totest), 416)


def test_pointcloud_limit_chaining(
    testpointcloud_mini: PointCloud, testpointcloud_mini_df
):
    totest = testpointcloud_mini.limit("x", minvalue=0.0, maxvalue=20000.0).limit(
        "x", 0.0, 2.0
    )
    np.testing.assert_array_equal(
        totest.data.values, testpointcloud_mini_df[0:2].values
    )


def test_pointcloud_limit_chaining2(testpointcloud_mini: PointCloud):
    totest = testpointcloud_mini.limit("x", minvalue=0.0, maxvalue=500.0).limit(
        "x", 0.0, 300.0
    )
    check.equal(len(totest), 4)
    check.less_equal(max(totest.data.x.values), 300.0)
    check.equal(totest.data.isnull().values.any(), False)


def test_limit_chaining3(testpointcloud_mini: PointCloud):
    totest = testpointcloud_mini.limit("x", minvalue=0.0, maxvalue=500.0)
    check.equal(totest.data.index.is_monotonic_increasing, True)
    check.equal(len(totest), 5)
