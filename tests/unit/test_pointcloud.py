from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_check as check
from pandas._testing import assert_frame_equal
from pyntcloud import PyntCloud

from pointcloudset import PointCloud


def test_init(testpointcloud_mini_df: pd.DataFrame):
    pointcloud = PointCloud(testpointcloud_mini_df, datetime(2020, 1, 1))
    check.equal(type(pointcloud), PointCloud)


def test_empty_pointcloud():
    empty_pc = PointCloud()
    check.equal(type(empty_pc), PointCloud)
    check.is_false(empty_pc._has_data())
    check.equal(len(empty_pc), 0)


def test_has_data(testpointcloud_mini: PointCloud):
    check.equal(testpointcloud_mini._has_data(), True)


def test_has_original_id(testpointcloud_mini: PointCloud):
    check.equal(testpointcloud_mini.has_original_id, False)


def test_has_original_id2(testpointcloud_mini_real: PointCloud):
    check.equal(testpointcloud_mini_real.has_original_id, True)


def test_contains_original_id_number(testpointcloud: PointCloud):
    check.equal(testpointcloud._contains_original_id_number(4700), True)
    check.equal(testpointcloud._contains_original_id_number(100000000), False)
    check.equal(testpointcloud._contains_original_id_number(1), False)
    check.equal(testpointcloud._contains_original_id_number(0), False)
    check.equal(testpointcloud._contains_original_id_number(-1000), False)


def test_points(testpointcloud_mini):
    points = testpointcloud_mini.points
    check.is_instance(points, PyntCloud)


def test_points2(testpointcloud_mini):
    points = testpointcloud_mini.points
    check.equal(
        list(points.points.columns),
        ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"],
    )


def test_timestamp(testpointcloud_mini):
    check.equal(type(testpointcloud_mini.timestamp), datetime)
    check.equal(testpointcloud_mini.timestamp, datetime(2020, 1, 1))


def test_timestamp2(testpointcloud_mini):
    fake_empty_df = pd.DataFrame.from_dict({"x": [0], "y": [0], "z": [0]})
    pc0 = PointCloud(data=fake_empty_df)
    pc1 = PointCloud(data=fake_empty_df)
    check.less(pc0.timestamp, pc1.timestamp)


def test_org_file(testpointcloud, testset):
    check.equal(type(testpointcloud.orig_file), str)
    check.equal(Path(testpointcloud.orig_file).stem, "test")


def test_len(testpointcloud_mini: PointCloud):
    check.equal(type(len(testpointcloud_mini)), int)
    check.equal(len(testpointcloud_mini), 8)


def test_getitem_single(testpointcloud: PointCloud):
    extracted = testpointcloud[0]
    check.equal(type(extracted), pd.DataFrame)
    check.equal(len(extracted), 1)
    check.equal(extracted.shape, (1, 10))
    check.equal(type(extracted.original_id.values[0]), np.uint32)
    check.equal(extracted.original_id.values[0], 4624)


def test_getitem_single_error(testpointcloud: PointCloud):
    with pytest.raises(IndexError):
        testpointcloud[1000000000]


def test_getitem_slice(testpointcloud: PointCloud):
    extracted = testpointcloud[0:3]
    check.equal(type(extracted), pd.DataFrame)
    check.equal(len(extracted), 3)
    check.equal(type(extracted.original_id.values[0]), np.uint32)
    check.equal(
        (extracted.original_id.values == np.array([4624, 4688, 4692])).all(), True
    )


def test_getitem_slice_full(testpointcloud: PointCloud):
    extracted = testpointcloud[:]
    check.equal(len(extracted), len(testpointcloud))
    check.equal(testpointcloud.data.equals(extracted), True)


def test_getitem_slice_step(testpointcloud: PointCloud):
    extracted = testpointcloud[0:10:2]
    check.equal(type(extracted), pd.DataFrame)
    check.equal(len(extracted), 5)
    check.equal(
        (
            extracted.original_id.values == np.array([4624, 4692, 4700, 4752, 4760])
        ).all(),
        True,
    )


def test_str(testpointcloud: PointCloud):
    check.equal(type(str(testpointcloud)), str)
    check.equal(
        str(testpointcloud),
        "pointcloud: with 45809 points, data:['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'original_id'], from Monday, June 22, 2020 01:40:42",
    )


def test_repr(testpointcloud_mini: PointCloud):
    check.equal(type(repr(testpointcloud_mini)), str)
    check.equal(len(repr(testpointcloud_mini)), 790)


def test_add_column(testpointcloud_mini: PointCloud):
    newframe = testpointcloud_mini._add_column("test", testpointcloud_mini.data["x"])
    check.equal(type(newframe), PointCloud)
    after_columns = list(testpointcloud_mini.data.columns.values)
    check.equal(
        str(after_columns),
        "['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'noise', 'range', 'test']",
    )


def test_describe(testpointcloud: PointCloud):
    check.equal(type(testpointcloud.describe()), pd.DataFrame)


# test with actual data
def test_testpointcloud_1_with_zero(testpointcloud_withzero: PointCloud):
    check.equal(len(testpointcloud_withzero), 131072)
    check.equal(testpointcloud_withzero._has_data(), True)


def test_testpointcloud_1(testpointcloud: PointCloud):
    # testpointcloud.data.to_pickle("/workspaces/pointcloudset/tests/testdata/testpointcloud_dataframe.pkl")
    check.equal(len(testpointcloud), 45809)
    check.equal(testpointcloud._has_data(), True)
    check.equal(len(testpointcloud), 45809)


def test_testpointcloud_index(testpointcloud):
    check.equal(testpointcloud.data.index[0], 0)
    check.equal(testpointcloud.data.index[-1] + 1, len(testpointcloud))
    check.equal(testpointcloud.data.index.is_monotonic_increasing, True)


def test_testpointcloud_2(testpointcloud_mini: PointCloud):
    check.equal(len(testpointcloud_mini), 8)
    check.equal(testpointcloud_mini._has_data(), True)


def test_testpointcloud_data(testpointcloud: PointCloud):
    data = testpointcloud.data
    check.equal(
        list(data.columns),
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
        ],
    )
    check.equal(data.shape, (45809, 10))


def test_testpointcloud_data_types(testpointcloud: PointCloud):
    types = [str(types) for types in testpointcloud.data.dtypes.values]
    check.equal(
        types,
        [
            "float32",
            "float32",
            "float32",
            "float32",
            "uint32",
            "uint16",
            "uint8",
            "uint16",
            "uint32",
            "uint32",
        ],
    )


def test_testpointcloud_withzero_data(
    testpointcloud_withzero: PointCloud,
    reference_data_with_zero_dataframe: pd.DataFrame,
):
    data = testpointcloud_withzero.data
    # data.to_pickle("/workspaces/pointcloudset/tests/testdata/testpointcloud_withzero_dataframe.pkl")
    check.equal(
        list(data.columns),
        ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"],
    )
    check.equal(data.shape, (131072, 9))
    assert_frame_equal(data, reference_data_with_zero_dataframe)


def test_testpointcloud_pointcloud(
    testpointcloud_withzero: PointCloud,
    reference_pointcloud_withzero_dataframe: pd.DataFrame,
):
    pointcloud = testpointcloud_withzero.to_instance("open3d")
    array = np.asarray(pointcloud.points)
    pointcloud_df = pd.DataFrame(array)
    # pointcloud_df.to_pickle(
    #    "/workspaces/pointcloudset/tests/testdata/testpointcloud_withzero_pointcloud.pkl"
    # )
    sub_points = pointcloud.select_by_index(list(range(5000, 5550)))
    sub_array = np.asarray(sub_points.points)

    check.equal(pointcloud.is_empty(), False)
    check.equal(np.sum(sub_array), 108.0019814982079)
    check.equal(np.sum(array), 60573.190267673606)
    assert_frame_equal(pointcloud_df, reference_pointcloud_withzero_dataframe)


def test_axis_aligned_bounding_box(testpointcloud_mini: PointCloud):
    bb = testpointcloud_mini.bounding_box
    check.is_instance(bb, pd.DataFrame)
    check.almost_equal(list(bb.x.values), [-1.0, 716.62253361])
    check.almost_equal(list(bb.y.values), [-1.0, 791.8389702424873])
    check.almost_equal(list(bb.z.values), [-1.0, 825.7276944386689])


def test_centroit(testpointcloud_mini: PointCloud):
    ct = testpointcloud_mini.centroid
    check.almost_equal(
        list(ct), [259.95131121217355, 225.64930989164827, 365.44029720089736]
    )
