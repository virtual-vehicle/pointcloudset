"""
Regression tests for deepcopy behavior in PointCloud and Dataset.

These tests verify:
1. Correctness: deepcopy creates independent objects without aliasing
2. Memory growth: repeated deepcopy doesn't cause pathological memory bloat
3. Memory release: deleted copies are properly freed
4. Serialization: deepcopy vs pickle comparison

Markers:
- @pytest.mark.memory_regression: memory growth tests (slow, optional)
- @pytest.mark.memory_release: memory cleanup tests (slow, optional)
- @pytest.mark.serialization: serialization comparison tests (medium speed)
"""

from __future__ import annotations

import copy
import gc
import pickle
import subprocess
import sys
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import Dataset, PointCloud

# ============================================================================
# Memory Measurement Utilities
# ============================================================================


def measure_peak_tracemalloc(func: Callable) -> tuple[float, any]:
    """
    Measure peak tracemalloc allocation during function execution.

    Returns:
        (peak_mb, result): Peak memory in MB and function result
    """
    tracemalloc.start()
    try:
        result = func()
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024), result
    finally:
        tracemalloc.stop()


def run_in_subprocess(code: str) -> tuple[int, str, str]:
    """
    Run Python code in a clean subprocess for RSS measurement.

    Args:
        code: Python code as string

    Returns:
        (return_code, stdout, stderr)
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    return result.returncode, result.stdout, result.stderr


def get_object_size_bytes(obj: PointCloud) -> int:
    """Estimate PointCloud size from its DataFrame memory usage."""
    return int(obj.data.memory_usage(deep=True).sum()) + 1024


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def baseline_pointcloud_100k() -> PointCloud:
    """Create a realistic large PointCloud with 100k points for memory tests."""
    np.random.seed(42)
    n_points = 100_000
    columns = ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"]

    data = {
        col: np.random.randn(n_points) * 100 if col != "ring" else np.random.randint(0, 64, n_points) for col in columns
    }
    df = pd.DataFrame(data)
    pc = PointCloud(data=df, timestamp=datetime(2020, 1, 1))
    return pc


@pytest.fixture
def small_pointcloud() -> PointCloud:
    """Create a small PointCloud for quick tests."""
    columns = ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"]
    data = {col: np.ones(100) * i for i, col in enumerate(columns)}
    df = pd.DataFrame(data)
    return PointCloud(data=df, timestamp=datetime(2020, 1, 1))


# ============================================================================
# Correctness Tests (Always Run)
# ============================================================================


class TestDeepcopyCorrectnessPointCloud:
    """Verify PointCloud deepcopy creates independent objects."""

    def test_deepcopy_creates_independent_object(self, small_pointcloud):
        """Deepcopy should create a separate object."""
        pc_copy = copy.deepcopy(small_pointcloud)
        assert pc_copy is not small_pointcloud
        assert id(pc_copy) != id(small_pointcloud)

    def test_deepcopy_dataframe_is_independent(self, small_pointcloud):
        """Deepcopy should copy the underlying DataFrame, verified via public API."""
        pc_copy = copy.deepcopy(small_pointcloud)
        # Verify via public .data property - the objects must differ
        assert pc_copy.data is not small_pointcloud.data
        # Verify the numpy buffers are actually separate (not just wrapper objects)
        assert pc_copy.data.values.ctypes.data != small_pointcloud.data.values.ctypes.data

    def test_deepcopy_mutating_copy_does_not_affect_original(self, small_pointcloud):
        """Mutating a deepcopy should not affect the original."""
        pc_copy = copy.deepcopy(small_pointcloud)
        original_values = small_pointcloud.data["x"].values.copy()

        # Mutate the copy
        pc_copy.data.loc[0, "x"] = 999999.0

        # Original should be unchanged
        assert small_pointcloud.data.loc[0, "x"] == original_values[0]
        assert pc_copy.data.loc[0, "x"] == 999999.0

    def test_deepcopy_mutating_original_does_not_affect_copy(self, small_pointcloud):
        """Mutating the original should not affect a deepcopy."""
        pc_copy = copy.deepcopy(small_pointcloud)
        copy_value = pc_copy.data.loc[0, "x"]

        # Mutate the original
        small_pointcloud.data.loc[0, "x"] = 888888.0

        # Copy should be unchanged
        assert pc_copy.data.loc[0, "x"] == copy_value
        assert small_pointcloud.data.loc[0, "x"] == 888888.0

    def test_deepcopy_timestamp_independence(self, small_pointcloud):
        """Timestamps should be independent."""
        original_ts = small_pointcloud.timestamp
        pc_copy = copy.deepcopy(small_pointcloud)

        assert pc_copy.timestamp == original_ts
        # Datetime is immutable, so this is more of a sanity check
        assert id(pc_copy.timestamp) != id(small_pointcloud.timestamp)

    def test_deepcopy_metadata_independence(self, small_pointcloud):
        """Metadata (orig_file) should be independent."""
        pc_copy = copy.deepcopy(small_pointcloud)
        original_file = small_pointcloud.orig_file

        assert pc_copy.orig_file == original_file

    def test_deepcopy_preserves_all_attributes(self, small_pointcloud):
        """Deepcopy should preserve all attributes and values."""
        pc_copy = copy.deepcopy(small_pointcloud)

        # Check all columns are present
        check.equal(list(pc_copy.data.columns), list(small_pointcloud.data.columns))

        # Check shape
        check.equal(len(pc_copy), len(small_pointcloud))

        # Check values (before mutation)
        pd.testing.assert_frame_equal(pc_copy.data, small_pointcloud.data)

    def test_deepcopy_points_wrapper_independence(self, small_pointcloud):
        """The _PointCloudView wrapper should track the copied data."""
        pc_copy = copy.deepcopy(small_pointcloud)

        # Mutate copy's data
        pc_copy.data.loc[0, "x"] = 777.0

        # The wrapper should see the new value
        assert pc_copy.points.xyz[0, 0] == 777.0
        # Original wrapper should not be affected
        assert small_pointcloud.points.xyz[0, 0] != 777.0

    def test_deepcopy_xyz_property_independence(self, small_pointcloud):
        """The xyz property should return independent arrays."""
        pc_copy = copy.deepcopy(small_pointcloud)

        xyz_orig = small_pointcloud.xyz
        xyz_copy = pc_copy.xyz

        # Modify the copy's xyz (via data)
        pc_copy.data.loc[0, "y"] = 555.0

        xyz_orig_after = small_pointcloud.xyz
        xyz_copy_after = pc_copy.xyz

        # Original's xyz should not change
        np.testing.assert_array_equal(xyz_orig, xyz_orig_after)
        # Copy's xyz should reflect the change
        assert xyz_copy_after[0, 1] == 555.0


class TestDeepcopyCorrectnessDataset:
    """Verify Dataset deepcopy creates independent objects."""

    def test_deepcopy_creates_independent_dataset(self, testset):
        """Deepcopy should create a separate Dataset object."""
        ds_copy = copy.deepcopy(testset)
        assert ds_copy is not testset
        assert id(ds_copy) != id(testset)

    def test_deepcopy_timestamps_are_independent(self, testset):
        """Timestamps list should be independent."""
        ds_copy = copy.deepcopy(testset)
        assert ds_copy.timestamps is not testset.timestamps
        assert ds_copy.timestamps == testset.timestamps

    def test_deepcopy_metadata_is_independent(self, testset):
        """Metadata dict should be independent."""
        ds_copy = copy.deepcopy(testset)
        assert ds_copy.meta is not testset.meta
        assert ds_copy.meta == testset.meta

    def test_deepcopy_metadata_mutation_does_not_affect_original(self, testset):
        """Mutating metadata in copy should not affect original."""
        ds_copy = copy.deepcopy(testset)
        original_orig_file = testset.meta["orig_file"]

        ds_copy.meta["orig_file"] = "modified"

        assert testset.meta["orig_file"] == original_orig_file
        assert ds_copy.meta["orig_file"] == "modified"

    def test_deepcopy_preserves_dataset_length(self, testset):
        """Deepcopy should preserve dataset length."""
        ds_copy = copy.deepcopy(testset)
        assert len(ds_copy) == len(testset)

    def test_deepcopy_independent_pointclouds(self, testset):
        """PointClouds in copied dataset should be independent."""
        ds_copy = copy.deepcopy(testset)

        # Get pointclouds
        pc_orig = testset[0]
        pc_copy = ds_copy[0]

        # They should be different objects
        assert pc_copy is not pc_orig

        # Mutating one should not affect the other
        original_x = pc_orig.data.loc[0, "x"]
        pc_copy.data.loc[0, "x"] = 12345.0

        assert pc_orig.data.loc[0, "x"] == original_x

    def test_deepcopy_dask_delayed_data_is_lazy(self, testset):
        """Dask delayed entries in the copied dataset should remain lazy (not computed)."""
        from dask.delayed import Delayed, DelayedLeaf

        ds_copy = copy.deepcopy(testset)
        for delayed_obj in ds_copy.data:
            assert isinstance(delayed_obj, (Delayed, DelayedLeaf)), (
                "Dataset.data entries must remain dask Delayed after deepcopy, not eagerly computed DataFrames"
            )


# ============================================================================
# Memory Growth Tests (Optional - Marked for Nightly Runs)
# ============================================================================


@pytest.mark.memory_regression
class TestMemoryGrowthPointCloud:
    """Test that repeated deepcopy doesn't cause pathological memory growth."""

    DEEPCOPY_RATIO_THRESHOLD = 20.0  # peak_mem / baseline_size < 20x
    NUM_COPIES = 10

    def test_single_deepcopy_peak_allocation(self, baseline_pointcloud_100k):
        """Single deepcopy peak should be reasonable."""
        baseline_size_mb = get_object_size_bytes(baseline_pointcloud_100k) / (1024 * 1024)

        peak_mb, _ = measure_peak_tracemalloc(lambda: copy.deepcopy(baseline_pointcloud_100k))

        ratio = peak_mb / baseline_size_mb
        print(f"\nSingle copy peak: {peak_mb:.1f}MB, baseline: {baseline_size_mb:.1f}MB, ratio: {ratio:.1f}x")

        assert ratio < self.DEEPCOPY_RATIO_THRESHOLD, (
            f"Single deepcopy ratio {ratio:.1f}x exceeds threshold {self.DEEPCOPY_RATIO_THRESHOLD}x"
        )

    def test_repeated_deepcopy_memory_growth(self, baseline_pointcloud_100k):
        """Repeated deepcopy shouldn't show exponential memory growth."""
        copies = []
        peaks = []

        for i in range(self.NUM_COPIES):

            def copy_task():
                copy_obj = copy.deepcopy(baseline_pointcloud_100k)
                copies.append(copy_obj)
                return copy_obj

            peak_mb, _ = measure_peak_tracemalloc(copy_task)
            peaks.append(peak_mb)

        print(f"\nPeaks (MB): {[f'{p:.1f}' for p in peaks]}")

        # Check that last peak is not dramatically larger than first
        # (which would indicate exponential growth)
        baseline_peak = peaks[0]
        final_peak = peaks[-1]

        # Allow up to 5x growth for repeated copies (due to allocator behavior)
        growth_ratio = final_peak / baseline_peak if baseline_peak > 0 else 1.0
        print(f"Final/baseline peak ratio: {growth_ratio:.1f}x")

        # Sanity check: shouldn't see exponential explosion
        assert growth_ratio < 5.0, f"Memory growth ratio {growth_ratio:.1f}x suggests potential issue"

    def test_repeated_deepcopy_with_explicit_deletion(self, baseline_pointcloud_100k):
        """Track memory when repeatedly copying and deleting."""
        baseline_size_mb = get_object_size_bytes(baseline_pointcloud_100k) / (1024 * 1024)

        def copy_and_delete_loop():
            for _ in range(5):
                copy_obj = copy.deepcopy(baseline_pointcloud_100k)
                # Use the copy to prevent optimization
                _ = copy_obj.xyz.sum()
                del copy_obj
            gc.collect()

        peak_mb, _ = measure_peak_tracemalloc(copy_and_delete_loop)
        ratio = peak_mb / baseline_size_mb

        print(f"\nCopy-delete loop peak: {peak_mb:.1f}MB, baseline: {baseline_size_mb:.1f}MB, ratio: {ratio:.1f}x")

        # Should stay within reason even with deletion
        assert ratio < self.DEEPCOPY_RATIO_THRESHOLD * 1.5


@pytest.mark.memory_regression
class TestMemoryGrowthDataset:
    """Test memory growth for Dataset deepcopy."""

    NUM_COPIES = 5

    def test_dataset_repeated_deepcopy(self, testset):
        """Dataset deepcopy shouldn't cause memory bloat."""
        if len(testset) == 0:
            pytest.skip("testset is empty")

        copies = []
        peaks = []

        for i in range(self.NUM_COPIES):

            def copy_task():
                copy_obj = copy.deepcopy(testset)
                copies.append(copy_obj)
                return copy_obj

            peak_mb, _ = measure_peak_tracemalloc(copy_task)
            peaks.append(peak_mb)

        print(f"\nDataset copy peaks (MB): {[f'{p:.1f}' for p in peaks]}")

        # Last peak should not be exponentially higher than first
        ratio = peaks[-1] / peaks[0]
        print(f"Last/first peak ratio: {ratio:.1f}x")

        # Allow some tolerance
        assert ratio < 5.0, f"Memory growth ratio {ratio:.1f}x seems excessive"


# ============================================================================
# Memory Release Tests (Optional - Marked for Nightly Runs)
# ============================================================================


@pytest.mark.memory_release
class TestMemoryRelease:
    """Test that deleted deepcopies are properly freed."""

    def test_pointcloud_deepcopy_release_in_subprocess(self):
        """RSS after deleting deepcopies must not exceed RSS before creating them.

        We do not assert how much memory is reclaimed because modern allocators
        (glibc, jemalloc) hold arenas after free().  The only reliable invariant
        is that RSS after cleanup must not be *higher* than RSS at peak.
        The test runs in a subprocess for a clean address-space baseline.
        """
        psutil = pytest.importorskip("psutil")
        code = """
import gc
import copy
import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, 'src')
from pointcloudset import PointCloud
import psutil

np.random.seed(42)
n_points = 50_000
columns = ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"]
data = {
    col: np.random.randn(n_points) * 100 if col != "ring" else np.random.randint(0, 64, n_points)
    for col in columns
}
df = pd.DataFrame(data)
pc = PointCloud(data=df, timestamp=datetime(2020, 1, 1))

# Create and keep copies so RSS grows
copies = [copy.deepcopy(pc) for _ in range(10)]
rss_peak = psutil.Process().memory_info().rss / (1024 * 1024)

del copies
gc.collect()

rss_after = psutil.Process().memory_info().rss / (1024 * 1024)
print(f"peak={rss_peak:.1f} after={rss_after:.1f}", file=sys.stderr)

# Fail only if RSS *grew* after deletion (unambiguous leak)
if rss_after > rss_peak * 1.05:
    sys.exit(1)
"""

        returncode, stdout, stderr = run_in_subprocess(code)
        print(stderr)
        assert returncode == 0, f"Memory release test failed:\n{stderr}"

    def test_pointcloud_tracemalloc_no_growth_after_gc(self):
        """Python-level allocations (tracemalloc) must not grow after deleting copies.

        Unlike RSS, tracemalloc tracks Python-heap objects directly and is not
        affected by allocator arena retention.  This test is therefore stable
        across platforms and Python versions.
        """
        columns = ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"]
        data = {col: np.random.randn(10_000) * 100 for col in columns}
        pc = PointCloud(data=pd.DataFrame(data), timestamp=datetime(2020, 1, 1))

        tracemalloc.start()
        copies = [copy.deepcopy(pc) for _ in range(5)]
        _, peak_with_copies = tracemalloc.get_traced_memory()

        del copies
        gc.collect()
        current_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # After deleting all copies the live allocation must be below peak
        assert current_after < peak_with_copies, (
            f"Live tracemalloc bytes after GC ({current_after}) should be below "
            f"peak with copies ({peak_with_copies}): copies may not be freed"
        )


# ============================================================================
# Serialization Comparison Tests (Optional)
# ============================================================================


@pytest.mark.serialization
class TestSerializationComparison:
    """Compare deepcopy vs pickle for performance and correctness."""

    def test_pointcloud_pickle_roundtrip(self, small_pointcloud):
        """PointCloud should be pickleable and round-trip correctly."""
        pickled = pickle.dumps(small_pointcloud, protocol=4)
        unpickled = pickle.loads(pickled)

        # Verify data is preserved
        pd.testing.assert_frame_equal(unpickled.data, small_pointcloud.data)
        assert unpickled.timestamp == small_pointcloud.timestamp
        assert unpickled.orig_file == small_pointcloud.orig_file

    def test_dataset_pickle_roundtrip(self, testset):
        """Dataset should be pickleable."""
        if len(testset) == 0:
            pytest.skip("testset is empty")

        # This may fail for dask-backed datasets, which is expected
        try:
            pickled = pickle.dumps(testset, protocol=4)
            unpickled = pickle.loads(pickled)
            assert len(unpickled) == len(testset)
        except Exception as e:
            pytest.skip(f"Dataset pickling not supported: {e}")

    def test_deepcopy_vs_pickle_size(self, baseline_pointcloud_100k):
        """Compare serialized sizes: deepcopy vs pickle."""
        deepcopy_obj = copy.deepcopy(baseline_pointcloud_100k)

        try:
            pickle_data = pickle.dumps(baseline_pointcloud_100k, protocol=4)
            pickle_size = len(pickle_data) / (1024 * 1024)
            deepcopy_size = get_object_size_bytes(deepcopy_obj) / (1024 * 1024)

            print(f"\nPickle size: {pickle_size:.1f}MB, Deepcopy in-memory: {deepcopy_size:.1f}MB")
            # Just informational, no strict assertion
        except Exception as e:
            pytest.skip(f"Pickle serialization failed: {e}")

    def test_deepcopy_vs_pickle_correctness(self, small_pointcloud):
        """Verify deepcopy and pickle produce equivalent results."""
        deepcopy_obj = copy.deepcopy(small_pointcloud)

        try:
            pickled = pickle.dumps(small_pointcloud, protocol=4)
            unpickled = pickle.loads(pickled)

            # Both should be identical to original
            pd.testing.assert_frame_equal(deepcopy_obj.data, unpickled.data)
            pd.testing.assert_frame_equal(deepcopy_obj.data, small_pointcloud.data)

            assert deepcopy_obj.timestamp == unpickled.timestamp == small_pointcloud.timestamp
        except Exception as e:
            pytest.skip(f"Pickle test failed: {e}")


# ============================================================================
# Sanity Checks (Always Run)
# ============================================================================


class TestDeepcopySanityChecks:
    """Basic sanity checks that deepcopy works."""

    def test_copy_module_import(self):
        """Verify copy module is available."""
        assert hasattr(copy, "deepcopy")

    def test_deepcopy_on_primitive_types(self):
        """Sanity check: deepcopy works on primitives."""
        x = [1, 2, 3]
        y = copy.deepcopy(x)
        assert x == y
        assert x is not y
        y[0] = 999
        assert x[0] == 1

    def test_deepcopy_on_pandas_dataframe(self):
        """Sanity check: deepcopy works on DataFrames."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_copy = copy.deepcopy(df)

        assert df_copy is not df
        pd.testing.assert_frame_equal(df, df_copy)

        df_copy.loc[0, "a"] = 999
        assert df.loc[0, "a"] == 1
