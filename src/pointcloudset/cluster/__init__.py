from __future__ import annotations

import numpy as np
import pandas
from scipy.spatial import KDTree

from pointcloudset.cluster.numba import roots_for_positions, union_pairs
from pointcloudset.config import (
    GET_CLUSTER_BORDER_QUERY_CHUNK_SIZE,
    GET_CLUSTER_CORE_QUERY_CHUNK_SIZE,
    GET_CLUSTER_MEMORY_BUDGET_MB,
)


def _budgeted_chunk_size(requested: int, max_neighbors: int, budget_bytes: int) -> int:
    if requested < 1:
        return 1
    if max_neighbors < 1:
        return requested

    # Conservative estimate for list-of-lists neighbour materialization.
    est_bytes_per_neighbor = 64
    est_list_overhead = 128
    safety_fraction = 0.60
    per_point_bytes = est_list_overhead + max_neighbors * est_bytes_per_neighbor
    usable_budget = max(1, int(budget_bytes * safety_fraction))
    max_points = max(1, usable_budget // max(1, per_point_bytes))
    return max(1, min(requested, max_points))


def get_cluster_labels(xyz: np.ndarray, eps: float, min_points: int) -> pandas.DataFrame:
    """Return DBSCAN labels for ``xyz`` coordinates.

    The implementation follows canonical DBSCAN with core connectivity via
    union-find and border attachment in a second pass.
    """
    n = len(xyz)
    tree = KDTree(xyz)

    # Stage 1: identify core points via count-only batch query (no edge storage).
    # Chunking keeps the count array allocation predictable for very large clouds.
    counts = np.empty(n, dtype=np.intp)
    chunk = max(1, min(n, 100_000))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        counts[start:end] = tree.query_ball_point(xyz[start:end], eps, workers=-1, return_length=True)
    is_core = counts >= min_points
    budget_bytes = int(GET_CLUSTER_MEMORY_BUDGET_MB * 1024 * 1024)

    if not is_core.any():
        return pandas.DataFrame(np.full(n, -1, dtype=np.intp), columns=["cluster"])

    # Stage 2: build core connectivity incrementally with union-find.
    # Keep DSU arrays only for core points to reduce memory and improve cache locality.
    core_idx = np.flatnonzero(is_core).astype(np.intp, copy=False)
    n_core = len(core_idx)

    core_pos = np.full(n, -1, dtype=np.intp)
    core_pos[core_idx] = np.arange(n_core, dtype=np.intp)

    parent = np.arange(n_core, dtype=np.intp)
    rank = np.zeros(n_core, dtype=np.uint8)

    # Query in chunks for throughput while keeping memory bounded.
    # Each chunk only materializes neighbour lists for that chunk.
    core_max_neighbors = int(counts[core_idx].max()) if n_core > 0 else 0
    edge_chunk_limit = _budgeted_chunk_size(GET_CLUSTER_CORE_QUERY_CHUNK_SIZE, core_max_neighbors, budget_bytes)
    edge_chunk = min(n_core, edge_chunk_limit)
    for start in range(0, n_core, edge_chunk):
        end = min(start + edge_chunk, n_core)
        batch_idx = core_idx[start:end]
        nbr_lists = tree.query_ball_point(xyz[batch_idx], eps, workers=-1)
        left_pairs: list[int] = []
        right_pairs: list[int] = []
        for local_i, nbrs in enumerate(nbr_lists):
            i_global = int(batch_idx[local_i])
            i_core = int(core_pos[i_global])
            for j_global in nbrs:
                if j_global <= i_global:
                    continue
                j_core = int(core_pos[j_global])
                if j_core >= 0:
                    left_pairs.append(i_core)
                    right_pairs.append(j_core)

        if left_pairs:
            left_arr = np.asarray(left_pairs, dtype=np.intp)
            right_arr = np.asarray(right_pairs, dtype=np.intp)
            union_pairs(parent, rank, left_arr, right_arr)

    # Stage 3: assign contiguous labels to connected core components.
    labels = np.full(n, -1, dtype=np.intp)
    core_positions = np.arange(n_core, dtype=np.intp)
    roots = roots_for_positions(parent, core_positions)
    _, inverse = np.unique(roots, return_inverse=True)
    labels[core_idx] = inverse.astype(np.intp, copy=False)

    # Stage 4: attach border points (non-core, but within eps of a core point).
    # Standard DBSCAN ambiguity: a border point reachable from multiple clusters
    # is assigned to the first encountered - matching open3d's behaviour.
    non_core_idx = np.flatnonzero(~is_core).astype(np.intp, copy=False)
    non_core_max_neighbors = int(counts[non_core_idx].max()) if len(non_core_idx) > 0 else 0
    border_chunk_limit = _budgeted_chunk_size(
        GET_CLUSTER_BORDER_QUERY_CHUNK_SIZE,
        non_core_max_neighbors,
        budget_bytes,
    )
    border_chunk = min(len(non_core_idx), border_chunk_limit) if len(non_core_idx) > 0 else 1
    for start in range(0, len(non_core_idx), border_chunk):
        end = min(start + border_chunk, len(non_core_idx))
        batch_idx = non_core_idx[start:end]
        nbr_lists = tree.query_ball_point(xyz[batch_idx], eps, workers=-1)
        for local_i, nbrs in enumerate(nbr_lists):
            i_global = int(batch_idx[local_i])
            for j_global in nbrs:
                if is_core[j_global]:
                    labels[i_global] = labels[int(j_global)]
                    break

    del core_pos

    return pandas.DataFrame(labels, columns=["cluster"])
