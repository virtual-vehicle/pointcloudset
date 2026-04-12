import numpy as np
import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import PointCloud
from pointcloudset.pointcloud_core import PointCloudCore


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


def test_plane_segmentation_of_open3d(testpointcloud):
    pcd = testpointcloud.limit("intensity", 500, 510).to_instance("open3d")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=500,
    )
    check.equal(len(inliers), 387)


def test_plane_segmentation(testpointcloud):
    plane = testpointcloud.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=False,
    )
    check.equal(type(plane), PointCloud)
    check.equal(len(plane), 387)


@pytest.mark.parametrize(
    "return_plane_model, type_out, for_len",
    [(True, dict, """plane["PointCloud"]"""), (False, PointCloud, """plane""")],
)
def test_plane_segmentation_plane_model(testpointcloud, return_plane_model, type_out, for_len):
    plane = testpointcloud.limit("intensity", 500, 510).plane_segmentation(
        distance_threshold=0.05,
        ransac_n=10,
        num_iterations=50,
        return_plane_model=return_plane_model,
    )
    check.equal(type(plane), type_out)
    check.equal(eval(f"len({for_len})"), 387)


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
    """Plane model for z=0 must have z-component dominant and intercept ~0."""
    pc, _ = _make_flat_plane_pc()
    result = pc.plane_segmentation(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        return_plane_model=True,
    )
    a, b, c, d = result["plane_model"]
    # For ax+by+cz+d=0 with plane z=0: |c| is the dominant normal component
    check.greater(abs(c), abs(a) + 1e-6)
    check.greater(abs(c), abs(b) + 1e-6)
    # Intercept d must be near zero (plane passes through origin)
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
