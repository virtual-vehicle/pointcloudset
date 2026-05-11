import resource

import numpy as np
import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import PointCloud


def _peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on Linux/macOS."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    if max_rss < 10_000_000:
        return int(max_rss * 1024)
    return int(max_rss)


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


# --- 300k synthetic fixture ---


@pytest.mark.slow
def test_cluster_300k_finds_10_clusters_within_1gib(testpointcloud_300k: PointCloud):
    """DBSCAN on the 300k fixture must find exactly 10 clusters within a 1 GiB peak-RSS increase.

    Fixture geometry:
        294k points in 10 Gaussian clusters (σ=0.9 m, centres ≥30 m apart in a 100³ m cube)
        6k uniform noise points

    Parameter rationale:
        eps=0.25 m  — roughly 1.8× the average inter-point spacing inside a dense cluster
                      (~0.14 m), so the cluster core chains reliably; at the same time it
                      keeps the total edge count under ~25M, putting the query_pairs array
                      well inside the 1 GiB budget. At eps=0.4 the edge count is ~173M
                      (~2.8 GB for the pairs array alone before any filtering).
        min_points=10 — far below the ~29 400 points per cluster, well above noise density.

    Cluster separation: all centres are ≥30 m apart; with eps=0.25 there is zero risk of
    two clusters bridging, so exactly 10 components are expected.
    """
    n_cluster_points = 294_000
    n_noise_points = 6_000
    n_centers = 10
    # Gaussian tails beyond ~3σ from each centre may be labelled noise; allow up to 2%
    # of cluster points to be cut off. Noise points far from all clusters remain noise.
    max_cluster_points_as_noise = int(n_cluster_points * 0.02)
    # A few noise points that land close to a cluster edge will be absorbed.
    max_noise_absorbed_into_cluster = n_noise_points // 10
    peak_before = _peak_rss_bytes()

    labels = testpointcloud_300k.get_cluster(eps=0.25, min_points=10)

    peak_after = _peak_rss_bytes()
    peak_delta = peak_after - peak_before

    unique_clusters = set(labels["cluster"].values) - {-1}
    n_noise = int((labels["cluster"] == -1).sum())
    n_clustered = len(labels) - n_noise

    check.equal(len(labels), len(testpointcloud_300k))
    # Exactly the right number of clusters — not merged, not split.
    check.equal(len(unique_clusters), n_centers)
    # Most cluster points must survive (cluster cores are dense relative to eps).
    check.greater(n_clustered, n_cluster_points - max_cluster_points_as_noise)
    # Most noise points must remain noise (uniform points at eps=0.25 have ~0 neighbours).
    check.less(n_noise, n_noise_points + max_noise_absorbed_into_cluster)
    assert peak_delta <= 1024**3, f"get_cluster peak RSS delta exceeded 1 GiB: {peak_delta / 1024**2:.1f} MiB"
