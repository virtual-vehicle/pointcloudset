import numpy as np
import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import PointCloud


def _make_synthetic_clustered_pc() -> PointCloud:
    """3 tight clusters (5 pts each) separated by ~10 units, plus 1 isolated noise point."""
    group_a = pd.DataFrame({"x": np.linspace(0.0, 0.04, 5), "y": np.zeros(5), "z": np.zeros(5)})
    group_b = pd.DataFrame({"x": np.linspace(10.0, 10.04, 5), "y": np.zeros(5), "z": np.zeros(5)})
    group_c = pd.DataFrame({"x": np.zeros(5), "y": np.linspace(10.0, 10.04, 5), "z": np.zeros(5)})
    noise = pd.DataFrame({"x": [100.0], "y": [100.0], "z": [0.0]})
    df = pd.concat([group_a, group_b, group_c, noise], ignore_index=True)
    return PointCloud(data=df)


def test_pointcloud_cluster(testpointcloud_mini_real: PointCloud):
    label = testpointcloud_mini_real.get_cluster(eps=0.8, min_points=5)
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


def test_pointcloud_take_cluster(testpointcloud_mini_real: PointCloud):
    label = testpointcloud_mini_real.get_cluster(eps=0.8, min_points=5)
    cluster0 = testpointcloud_mini_real.take_cluster(0, label)
    check.equal(len(cluster0), 5)
    check.equal(type(cluster0), PointCloud)
    check.equal(testpointcloud_mini_real.timestamp, cluster0.timestamp)
    check.equal(cluster0.data.index.is_monotonic_increasing, True)


# --- Synthetic behavioral tests (implementation-agnostic) ---


def test_get_cluster_returns_dataframe_with_cluster_column():
    pc = _make_synthetic_clustered_pc()
    labels = pc.get_cluster(eps=0.1, min_points=3)
    check.is_instance(labels, pd.DataFrame)
    check.equal(list(labels.columns), ["cluster"])
    check.equal(len(labels), len(pc))


def test_get_cluster_finds_three_clusters():
    """Well-separated groups must yield exactly 3 non-noise cluster IDs."""
    pc = _make_synthetic_clustered_pc()
    labels = pc.get_cluster(eps=0.1, min_points=3)
    non_noise = set(labels["cluster"].values) - {-1}
    check.equal(len(non_noise), 3)


def test_get_cluster_isolated_point_is_noise():
    """The isolated point (last row) must receive label -1."""
    pc = _make_synthetic_clustered_pc()
    labels = pc.get_cluster(eps=0.1, min_points=3)
    check.equal(int(labels["cluster"].iloc[-1]), -1)


def test_get_cluster_same_group_same_label():
    """Points within the same spatial group must share one label; groups differ."""
    pc = _make_synthetic_clustered_pc()
    labels = pc.get_cluster(eps=0.1, min_points=3)
    arr = labels["cluster"].values
    # Each group's 5 points → all identical label
    check.equal(len(set(arr[0:5])), 1)
    check.equal(len(set(arr[5:10])), 1)
    check.equal(len(set(arr[10:15])), 1)
    # The three groups carry distinct labels
    check.not_equal(arr[0], arr[5])
    check.not_equal(arr[0], arr[10])
    check.not_equal(arr[5], arr[10])


# --- Input validation ---


def test_get_cluster_raises_on_zero_eps():
    pc = _make_synthetic_clustered_pc()
    with pytest.raises(ValueError, match="eps"):
        pc.get_cluster(eps=0.0, min_points=3)


def test_get_cluster_raises_on_negative_eps():
    pc = _make_synthetic_clustered_pc()
    with pytest.raises(ValueError, match="eps"):
        pc.get_cluster(eps=-1.0, min_points=3)


def test_get_cluster_raises_on_zero_min_points():
    pc = _make_synthetic_clustered_pc()
    with pytest.raises(ValueError, match="min_points"):
        pc.get_cluster(eps=0.1, min_points=0)


def test_get_cluster_raises_on_empty_pointcloud():
    empty = PointCloud(data=pd.DataFrame({"x": [], "y": [], "z": []}))
    with pytest.raises(ValueError, match="empty"):
        empty.get_cluster(eps=0.1, min_points=3)


# --- Noise cluster (label -1) ---


def test_take_cluster_noise_points():
    """take_cluster(-1) must return the isolated noise point."""
    pc = _make_synthetic_clustered_pc()
    labels = pc.get_cluster(eps=0.1, min_points=3)
    noise = pc.take_cluster(-1, labels)
    check.is_instance(noise, PointCloud)
    check.equal(len(noise), 1)
    check.almost_equal(float(noise.data["x"].iloc[0]), 100.0, 3)


# --- Mismatched labels guard ---


def test_take_cluster_raises_on_length_mismatch():
    pc = _make_synthetic_clustered_pc()
    labels = pc.get_cluster(eps=0.1, min_points=3)
    short_labels = labels.iloc[:-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="cluster_labels"):
        pc.take_cluster(0, short_labels)
