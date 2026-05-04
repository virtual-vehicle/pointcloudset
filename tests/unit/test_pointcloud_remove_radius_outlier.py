import numpy as np
import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import PointCloud


def test_rro1(testpointcloud_mini_real):
    test = testpointcloud_mini_real.filter("radiusoutlier", nb_points=500, radius=0.1)
    check.equal(test._has_data(), False)


# --- Synthetic behavioral tests (implementation-agnostic) ---


def _make_cluster_plus_outlier_pc(n_cluster: int = 20) -> PointCloud:
    """Dense cluster near origin plus one isolated point at (100, 100, 0)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.uniform(-0.1, 0.1, n_cluster), [100.0]]),
            "y": np.concatenate([rng.uniform(-0.1, 0.1, n_cluster), [100.0]]),
            "z": np.zeros(n_cluster + 1),
        }
    )
    return PointCloud(data=df)


def test_rro_removes_isolated_point():
    """Isolated point must be removed while the dense cluster is kept."""
    n_cluster = 20
    pc = _make_cluster_plus_outlier_pc(n_cluster)
    result = pc.filter("radiusoutlier", nb_points=5, radius=1.0)
    check.equal(len(result), n_cluster)
    # All surviving points belong to the dense cluster near the origin
    check.is_true(np.all(np.abs(result.data["x"].values) < 0.2))
    check.is_true(np.all(np.abs(result.data["y"].values) < 0.2))


def test_rro_all_points_removed_when_all_isolated():
    """Every point is isolated → all should be removed."""
    df = pd.DataFrame(
        {
            "x": [0.0, 10.0, 20.0, 30.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pc = PointCloud(data=df)
    result = pc.filter("radiusoutlier", nb_points=2, radius=1.0)
    check.equal(result._has_data(), False)


def test_rro_all_points_kept_when_all_dense():
    """All tightly packed points must survive."""
    n = 10
    df = pd.DataFrame(
        {
            "x": np.linspace(0.0, 0.05, n),
            "y": np.zeros(n),
            "z": np.zeros(n),
        }
    )
    pc = PointCloud(data=df)
    result = pc.filter("radiusoutlier", nb_points=5, radius=1.0)
    check.equal(len(result), n)


def test_rro_result_is_subset_of_input():
    """The filter result must always be a subset of the input."""
    rng = np.random.default_rng(123)
    n = 50
    df = pd.DataFrame(
        {
            "x": rng.uniform(-10, 10, n),
            "y": rng.uniform(-10, 10, n),
            "z": rng.uniform(-10, 10, n),
        }
    )
    pc = PointCloud(data=df)
    result = pc.filter("radiusoutlier", nb_points=3, radius=5.0)
    check.less_equal(len(result), n)
    if result._has_data():
        check.is_true(result.data["x"].isin(pc.data["x"]).all())


def test_rro_preserves_pointcloud_type():
    """Return type must always be PointCloud."""
    pc = _make_cluster_plus_outlier_pc()
    result = pc.filter("radiusoutlier", nb_points=5, radius=1.0)
    check.is_instance(result, PointCloud)


# --- Input validation ---


def test_rro_raises_on_zero_nb_points():
    pc = _make_cluster_plus_outlier_pc()
    with pytest.raises(ValueError, match="nb_points"):
        pc.filter("radiusoutlier", nb_points=0, radius=1.0)


def test_rro_raises_on_negative_nb_points():
    pc = _make_cluster_plus_outlier_pc()
    with pytest.raises(ValueError, match="nb_points"):
        pc.filter("radiusoutlier", nb_points=-1, radius=1.0)


def test_rro_raises_on_zero_radius():
    pc = _make_cluster_plus_outlier_pc()
    with pytest.raises(ValueError, match="radius"):
        pc.filter("radiusoutlier", nb_points=5, radius=0.0)


def test_rro_raises_on_negative_radius():
    pc = _make_cluster_plus_outlier_pc()
    with pytest.raises(ValueError, match="radius"):
        pc.filter("radiusoutlier", nb_points=5, radius=-1.0)


# --- Edge cases ---


def test_rro_empty_pointcloud_returns_empty():
    empty = PointCloud(data=pd.DataFrame({"x": [], "y": [], "z": []}))
    result = empty.filter("radiusoutlier", nb_points=1, radius=1.0)
    check.is_instance(result, PointCloud)
    check.equal(result._has_data(), False)


def test_rro_single_point_is_removed():
    """A single point has no neighbours, so it must always be removed."""
    pc = PointCloud(data=pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))
    result = pc.filter("radiusoutlier", nb_points=1, radius=1.0)
    check.equal(result._has_data(), False)
