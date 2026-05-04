import numpy as np
import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import PointCloud


def _make_flat_plane_pc(n_inliers: int = 80, n_outliers: int = 10) -> tuple[PointCloud, int]:
    """Points on z=0 plane (inliers) plus points at z=5 (outliers)."""
    rng = np.random.default_rng(42)
    inlier_df = pd.DataFrame(
        {
            "x": rng.uniform(-5, 5, n_inliers),
            "y": rng.uniform(-5, 5, n_inliers),
            "z": np.zeros(n_inliers),
        }
    )
    outlier_df = pd.DataFrame(
        {
            "x": rng.uniform(-5, 5, n_outliers),
            "y": rng.uniform(-5, 5, n_outliers),
            "z": np.full(n_outliers, 5.0),
        }
    )
    df = pd.concat([inlier_df, outlier_df], ignore_index=True)
    return PointCloud(data=df), n_inliers


def test_plane_segmentation(testpointcloud):
    plane = testpointcloud.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=False,
    )
    check.equal(type(plane), PointCloud)
    check.greater(len(plane), 0)


def test_plane_segmentation_returns_pointcloud(testpointcloud):
    plane = testpointcloud.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=False,
    )
    check.is_instance(plane, PointCloud)


def test_plane_segmentation_returns_dict_with_model(testpointcloud):
    result = testpointcloud.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=True,
    )
    check.is_instance(result, dict)
    check.is_in("PointCloud", result)
    check.is_in("plane_model", result)
    check.is_instance(result["PointCloud"], PointCloud)
    check.equal(len(result["plane_model"]), 4)
    check.greater(len(result["PointCloud"]), 0)


# --- Synthetic behavioral tests (implementation-agnostic) ---


def test_plane_segmentation_synthetic_inlier_count():
    """RANSAC on a perfect z=0 plane must recover all inliers."""
    pc, n_inliers = _make_flat_plane_pc()
    result = pc.plane_segmentation(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        return_plane_model=False,
    )
    check.equal(type(result), PointCloud)
    check.equal(len(result), n_inliers)


def test_plane_segmentation_synthetic_returns_correct_types():
    """With return_plane_model=True the result is a dict with expected keys."""
    pc, n_inliers = _make_flat_plane_pc()
    result = pc.plane_segmentation(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        return_plane_model=True,
    )
    check.is_instance(result, dict)
    check.is_in("plane_model", result)
    check.is_in("PointCloud", result)
    check.equal(len(result["PointCloud"]), n_inliers)


def test_plane_segmentation_synthetic_plane_model_normal():
    """Refit plane model for z=0 must have z-component dominant and intercept ~0."""
    pc, _ = _make_flat_plane_pc()
    result = pc.plane_segmentation(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        return_plane_model=True,
    )
    a, b, c, d = result["plane_model"]
    check.greater(abs(c), abs(a) + 1e-6)
    check.greater(abs(c), abs(b) + 1e-6)
    check.almost_equal(float(d), 0.0, 2)


def test_plane_segmentation_result_only_contains_inliers():
    """All points in the result must lie on the fitted plane (z ≈ 0)."""
    pc, _ = _make_flat_plane_pc()
    result = pc.plane_segmentation(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
    )
    check.is_true(np.allclose(result.data["z"].values, 0.0, atol=0.01))


def test_plane_segmentation_seed_is_reproducible():
    """Same seed must produce identical results across two calls."""
    pc, _ = _make_flat_plane_pc()
    kwargs = dict(distance_threshold=0.01, ransac_n=3, num_iterations=50, return_plane_model=True)
    r1 = pc.plane_segmentation(**kwargs, seed=7)
    r2 = pc.plane_segmentation(**kwargs, seed=7)
    np.testing.assert_array_equal(r1["plane_model"], r2["plane_model"])
    check.equal(len(r1["PointCloud"]), len(r2["PointCloud"]))


def test_plane_segmentation_different_seeds_may_differ():
    """Different seeds should be accepted without error."""
    pc, _ = _make_flat_plane_pc()
    r1 = pc.plane_segmentation(distance_threshold=0.01, ransac_n=3, num_iterations=10, seed=1)
    r2 = pc.plane_segmentation(distance_threshold=0.01, ransac_n=3, num_iterations=10, seed=999)
    check.is_instance(r1, PointCloud)
    check.is_instance(r2, PointCloud)


# --- Input validation ---


def test_plane_segmentation_raises_on_empty_pointcloud():
    empty = PointCloud(data=pd.DataFrame({"x": [], "y": [], "z": []}))
    with pytest.raises(ValueError, match="empty"):
        empty.plane_segmentation(distance_threshold=0.1, ransac_n=3, num_iterations=10)


def test_plane_segmentation_raises_on_zero_distance_threshold():
    pc, _ = _make_flat_plane_pc()
    with pytest.raises(ValueError, match="distance_threshold"):
        pc.plane_segmentation(distance_threshold=0.0, ransac_n=3, num_iterations=10)


def test_plane_segmentation_raises_on_negative_distance_threshold():
    pc, _ = _make_flat_plane_pc()
    with pytest.raises(ValueError, match="distance_threshold"):
        pc.plane_segmentation(distance_threshold=-1.0, ransac_n=3, num_iterations=10)


def test_plane_segmentation_raises_on_ransac_n_less_than_3():
    pc, _ = _make_flat_plane_pc()
    with pytest.raises(ValueError, match="ransac_n"):
        pc.plane_segmentation(distance_threshold=0.1, ransac_n=2, num_iterations=10)


def test_plane_segmentation_raises_on_ransac_n_exceeds_points():
    pc, _ = _make_flat_plane_pc(n_inliers=5, n_outliers=0)
    with pytest.raises(ValueError, match="ransac_n"):
        pc.plane_segmentation(distance_threshold=0.1, ransac_n=10, num_iterations=10)


def test_plane_segmentation_raises_on_zero_iterations():
    pc, _ = _make_flat_plane_pc()
    with pytest.raises(ValueError, match="num_iterations"):
        pc.plane_segmentation(distance_threshold=0.1, ransac_n=3, num_iterations=0)
