import datetime

import numpy as np
import pandas as pd
import pytest
import pytest_check as check
from dask.delayed import DelayedLeaf
from pandas.testing import assert_series_equal

from pointcloudset import Dataset, PointCloud


@pytest.mark.parametrize("test_sets", ["testset", "testdataset_vz6000"], indirect=True)
def test_getitem(test_sets: Dataset):
    check.equal(type(test_sets[0]), PointCloud)


@pytest.mark.parametrize("test_sets", ["testset", "testdataset_vz6000"], indirect=True)
def test_getitem_late(test_sets):
    check.equal(type(test_sets[1]), PointCloud)


@pytest.mark.parametrize("test_sets", ["testset", "testdataset_vz6000"], indirect=True)
def test_getitem_2times(test_sets):
    check.equal(type(test_sets[0]), PointCloud)
    check.equal(type(test_sets[1]), PointCloud)
    check.equal(type(test_sets[0]), PointCloud)
    check.equal(type(test_sets[1]), PointCloud)


@pytest.mark.parametrize("test_sets", ["testset", "testdataset_vz6000"], indirect=True)
def test_getitem_slice(test_sets: Dataset):
    test = test_sets[0:2]
    testlist = [0, 1, 2, 3]
    check.is_instance(test, Dataset)
    check.equal(len(test), 2)
    check.equal(len(test), len(testlist[0:2]))
    check.equal(type(test_sets[0:2][0]), PointCloud)


@pytest.mark.parametrize("test_sets", ["testset", "testdataset_vz6000"], indirect=True)
def test_getitem_error(test_sets: Dataset):
    with pytest.raises(TypeError):
        test_sets["fake"]


@pytest.mark.parametrize("test_sets", ["testset", "testdataset_vz6000"], indirect=True)
def test_get_pointcloud_number_from_time(test_sets):
    res = test_sets._get_pointcloud_number_from_time(test_sets.start_time)
    check.equal(res, 0)
    res2 = test_sets._get_pointcloud_number_from_time(test_sets.end_time)
    check.equal(res2 + 1, len(test_sets))


def test_getitem_timerange(testset: Dataset):
    check.equal(len(testset), 2)
    dataset = testset.get_pointclouds_between_timestamps(
        datetime.datetime(2020, 6, 22, 13, 40, 42, 657267),
        datetime.datetime(2020, 6, 22, 13, 40, 42, 755912),
    )
    check.equal(len(dataset), 2)
    check.equal(type(dataset[0:2]), Dataset)
    check.equal(type(dataset[0:2][0]), PointCloud)


def test_getitem_strange(testset):
    check.equal(len(testset), 2)
    check.equal(len(testset[2:0]), 0)


@pytest.mark.parametrize("test_sets", ["testset", "testdataset_vz6000"], indirect=True)
def test_has_pointclouds(test_sets: Dataset):
    check.equal(test_sets.has_pointclouds(), True)


def test_str(testset: Dataset):
    check.equal(type(str(testset)), str)
    check.equal(
        str(testset),
        "Lidar Dataset with 2 pointcloud(s)",
    )


def test_repr(testset: Dataset):
    repr = testset.__repr__()
    check.equal(type(repr), str)


def test_start_time(testset: Dataset):
    st = testset.start_time
    check.is_instance(st, datetime.datetime)
    check.equal(st, datetime.datetime(2020, 6, 22, 13, 40, 42, 657267))


def test_end_time(testset: Dataset):
    et = testset.end_time
    check.is_instance(et, datetime.datetime)
    check.equal(et, datetime.datetime(2020, 6, 22, 13, 40, 42, 755912))


@pytest.mark.parametrize("test_sets", ["testset", "testdataset_vz6000"], indirect=True)
def test_time(test_sets: Dataset):
    check.greater(test_sets.end_time, test_sets.start_time)


def test_extend(testbag1):
    ds1 = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    len_old = len(ds1)
    ds2 = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    ds2.timestamps = [ts + datetime.timedelta(seconds=10) for ts in ds2.timestamps]
    ds1.extend(ds2)
    check.equal(len(ds1), len_old * 2)


def test_from_pointclouds_list(testdataset_mini_real):
    check.is_instance(testdataset_mini_real, Dataset)
    check.equal(len(testdataset_mini_real), 2)
    check.is_instance(testdataset_mini_real.data[0], DelayedLeaf)
    check.is_instance(testdataset_mini_real[0], PointCloud)


@pytest.fixture()
def expected_columns():
    return [
        "x min",
        "y min",
        "z min",
        "intensity min",
        "t min",
        "reflectivity min",
        "ring min",
        "noise min",
        "range min",
        "timestamp",
    ]


def test_agg_pointcloud(
    testdataset_mini_real: Dataset,
    testpointcloud_mini_real: PointCloud,
    expected_columns,
):
    test = testdataset_mini_real._agg_per_pointcloud("min")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(len(test), len(testdataset_mini_real))
    check.equal(list(test.columns), expected_columns)
    check.equal(test.min()["x min"], x_min.values[0])


def test_agg_pointcloud(
    testdataset_mini_real: Dataset,
    testpointcloud_mini_real: PointCloud,
    expected_columns,
):
    test = testdataset_mini_real.agg("min", "pointcloud")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(len(test), len(testdataset_mini_real))
    check.equal(
        list(test.columns),
        expected_columns,
    )
    check.equal(test.min()["x min"], x_min.values[0])


def test_agg_dict_pointcloud(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    test = testdataset_mini_real.agg({"x": "min"}, "pointcloud")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, list)
    check.equal(len(test), len(testdataset_mini_real))
    check.equal(test[0].values[0], x_min.values[0])


def test_agg_list1(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    f0 = testdataset_mini_real[0].data.agg(["min", "max"])
    f1 = testdataset_mini_real[1].data.agg(["min", "max"])
    test = testdataset_mini_real.agg(["min", "max"], "pointcloud")
    check.is_instance(test, list)


def test_agg_dataset(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    test = testdataset_mini_real.agg("min", "dataset")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.Series)
    check.equal(
        list(test.index),
        [
            "x min",
            "y min",
            "z min",
            "intensity min",
            "t min",
            "reflectivity min",
            "ring min",
            "noise min",
            "range min",
            "N min",
            "original_id min",
        ],
    )
    check.equal(test["x min"], x_min.values[0])


def test_agg_dataset_dict_dataset(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    test = testdataset_mini_real.agg({"x": "min"}, "dataset")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.Series)
    check.equal(
        list(test.index),
        [
            "x {'x': 'min'}",
        ],
    )
    check.equal(test.values[0], x_min.values[0])


def test_agg_dataset_list(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    test = testdataset_mini_real.agg(["min", "max"], "dataset")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(
        list(test.index),
        ["min", "max"],
    )
    check.equal(test.x["min"]["min"], x_min.values[0])


def test_agg_list_pointcoud(testdataset_mini_real: Dataset):
    test = testdataset_mini_real.agg(["min", "max"], "pointcloud")
    check.is_instance(test, list)


def test_agg_point(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    test = testdataset_mini_real.agg("min", "point")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(
        list(test.columns),
        [
            "x min",
            "y min",
            "z min",
            "intensity min",
            "t min",
            "reflectivity min",
            "ring min",
            "noise min",
            "range min",
            "N",
            "original_id",
        ],
    )
    check.equal(test.min()["x min"], x_min.values[0])


def test_agg_point_list(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    test = testdataset_mini_real.agg(["min", "max"], "point")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(test.x["min"].min(), x_min.values[0])


def test_agg_point_dict(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    test = testdataset_mini_real.agg({"x": "min"}, "point")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(
        list(test.columns),
        ["x", "N", "original_id"],
    )
    check.equal(test.min()["x"], x_min.values[0])


def test_dataset_min_dataset(
    testdataset_mini_real: Dataset,
    testpointcloud_mini_real: PointCloud,
    testpointcloud_mini_real_plus1: PointCloud,
):
    mincalc = testdataset_mini_real.min(depth="dataset")
    minshould = testpointcloud_mini_real.data.min()
    minshould.index = [f"{i} min" for i in minshould.index]
    check.is_instance(mincalc, pd.Series)
    assert_series_equal(
        mincalc.drop("N min"),
        minshould,
        check_names=False,
    )


def test_dataset_min_point(
    testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud
):
    mincalc = testdataset_mini_real.min(depth="point")
    first = mincalc.drop(["N", "original_id"], axis=1).iloc[0]
    check.equal(mincalc.iloc[0].N, 2)
    check.is_instance(mincalc, pd.DataFrame)
    should = (
        testpointcloud_mini_real.extract_point(6008, use_original_id=True)
        .drop(["original_id"], axis=1)
        .squeeze()
    )
    should.index = [f"{i} min" for i in should.index]
    assert_series_equal(first, should, check_names=False)


def test_dataset_min_pointcloud(
    testdataset_mini_real: Dataset,
    testpointcloud_mini_real: PointCloud,
):
    mincalc = testdataset_mini_real.min(depth="pointcloud")
    x_min = testpointcloud_mini_real.data.agg({"x": "min"})
    check.equal(mincalc["x min"][0], x_min.values[0])


def test_dataset_std(testdataset_mini_same: Dataset):
    stdres = testdataset_mini_same.std()
    check.is_instance(stdres, pd.Series)
    check.equal(stdres["x std"], 0.0)


def test_pointcloud_std(testdataset_mini_real: Dataset):
    stdres = testdataset_mini_real.std("pointcloud")
    check.is_instance(stdres, pd.DataFrame)
    check.equal(len(stdres), 2)
    check.equal(
        list(stdres.columns),
        [
            "x std",
            "y std",
            "z std",
            "intensity std",
            "t std",
            "reflectivity std",
            "ring std",
            "noise std",
            "range std",
            "timestamp",
        ],
    )


def test_pointcloud_point(testdataset_mini_real: Dataset):
    stdres = testdataset_mini_real.std("point")
    check.is_instance(stdres, pd.DataFrame)
    check.equal(
        list(stdres.columns),
        [
            "x std",
            "y std",
            "z std",
            "intensity std",
            "t std",
            "reflectivity std",
            "ring std",
            "noise std",
            "range std",
            "N",
            "original_id",
        ],
    )
    check.less(abs((stdres["x std"].values - 0.707107)).max(), 1e-6)


def test_dataset_max_dataset(
    testdataset_mini_real: Dataset,
    testpointcloud_mini_real: PointCloud,
    testpointcloud_mini_real_plus1: PointCloud,
):
    calc = testdataset_mini_real.max(depth="dataset")
    should = testpointcloud_mini_real_plus1.data.max()
    should.index = [f"{i} max" for i in should.index]
    check.is_instance(calc, pd.Series)
    assert_series_equal(
        calc.drop("N max"),
        should,
        check_names=False,
    )


def test_dataset_max_pointcloud(
    testdataset_mini_real: Dataset,
    testpointcloud_mini_real: PointCloud,
    testpointcloud_mini_real_plus1: PointCloud,
):
    calc = testdataset_mini_real.max(depth="pointcloud")
    x_max0 = testpointcloud_mini_real.data.agg({"x": "max"})
    x_max1 = testpointcloud_mini_real_plus1.data.agg({"x": "max"})
    check.equal(calc["x max"][0], x_max0.values[0])
    check.equal(calc["x max"][1], x_max1.values[0])


def test_dataset_max_point(
    testdataset_mini_real: Dataset,
    testpointcloud_mini_real: PointCloud,
    testpointcloud_mini_real_plus1: PointCloud,
):
    calc = testdataset_mini_real.max(depth="point")
    first = calc.drop(["N", "original_id"], axis=1).iloc[0]
    check.equal(calc.iloc[0].N, 2)
    check.is_instance(calc, pd.DataFrame)
    should = (
        testpointcloud_mini_real_plus1.extract_point(6008, use_original_id=True)
        .drop(["original_id"], axis=1)
        .squeeze()
    )
    should.index = [f"{i} max" for i in should.index]
    assert_series_equal(first, should, check_names=False)


def test_dataset_mean(
    testdataset_mini_same: Dataset, testpointcloud_mini_real: PointCloud
):
    calc = testdataset_mini_same.mean()
    should = testpointcloud_mini_real.data.mean()
    check.is_instance(calc, pd.Series)
    should.index = [f"{i} mean" for i in should.index]
    check.is_instance(calc, pd.Series)
    assert_series_equal(
        calc.drop("N mean"),
        should,
        check_names=False,
    )


empty_data = pd.DataFrame.from_dict(
    {
        "x": {0: 0.0},
        "y": {0: -0.0},
        "z": {0: 0.0},
        "intensity": {0: 0.0},
        "t": {0: 0.0},
        "reflectivity": {0: 0.0},
        "ring": {0: 0.0},
        "noise": {0: 74.0},
        "range": {0: 0.0},
    }
)


def test_replace_empty_frames_with_nan(testdataset_with_empty_frame: Dataset):
    test = testdataset_with_empty_frame._replace_empty_frames_with_nan(
        empty_data=empty_data
    )
    check.equal(len(test), len(testdataset_with_empty_frame))
    check.equal(len(test[1]), 1)


def test_replace_nan_frames_with_empty(testdataset_with_empty_frame: Dataset):
    test = testdataset_with_empty_frame._replace_empty_frames_with_nan(
        empty_data=empty_data
    )
    test2 = test._replace_nan_frames_with_empty(empty_data=empty_data)
    check.equal(len(test2), len(testdataset_with_empty_frame))
    check.is_false(test2[1]._has_data())
    check.equal(len(test2[1]), 0)


def test_all_have_origianl_ids(testset: Dataset, testdataset_vz6000: Dataset):
    check.is_true(testset.has_original_id)
    check.is_false(testdataset_vz6000.has_original_id)


def test_dataset_vz6000_min_dataset(
    testdataset_vz6000: Dataset, testvz6000_1: PointCloud, testvz6000_2: PointCloud
):
    res = testdataset_vz6000.min(depth="dataset")
    min_1 = testvz6000_1.data.min()
    min_2 = testvz6000_2.data.min()
    check.is_instance(res, pd.Series)
    check.equal(res["intensity min"], min(min_1["intensity"], min_2["intensity"]))


def test_dataset_vz6000_max_dataset(
    testdataset_vz6000: Dataset, testvz6000_1: PointCloud, testvz6000_2: PointCloud
):
    res = testdataset_vz6000.max(depth="dataset")
    max_1 = testvz6000_1.data.max()
    max_2 = testvz6000_2.data.max()
    check.is_instance(res, pd.Series)
    check.equal(res["intensity max"], max(max_1["intensity"], max_2["intensity"]))


def test_dataset_vz6000_max_pointcloud(
    testdataset_vz6000: Dataset, testvz6000_1: PointCloud, testvz6000_2: PointCloud
):
    res = testdataset_vz6000.max(depth="pointcloud")
    check.is_instance(res, pd.DataFrame)
    max_1 = testvz6000_1.data.max().x
    max_2 = testvz6000_2.data.max().x
    np.testing.assert_array_equal(res["x max"].values, np.array([max_1, max_2]))


def test_dataset_vz6000_agg_point(testdataset_vz6000: Dataset):
    with pytest.raises(ValueError):
        testdataset_vz6000.min(depth="point")


def test_dataset_bounding_box(
    testdataset_mini_real: Dataset,
    testpointcloud_mini_real: PointCloud,
    testpointcloud_mini_real_plus1: PointCloud,
):
    bb = testdataset_mini_real.bounding_box
    min_1 = testpointcloud_mini_real.data.min()[["x", "y", "z"]]
    max_2 = testpointcloud_mini_real_plus1.data.max()[["x", "y", "z"]]
    check.is_instance(bb, pd.DataFrame)
    pd.testing.assert_series_equal(
        bb.iloc[0], min_1, check_dtype=False, check_names=False
    )
    pd.testing.assert_series_equal(
        bb.iloc[1], max_2, check_dtype=False, check_names=False
    )


def test_dataset_agg_long(testset: Dataset):
    res = testset.agg(
        {
            "x": ["min", "max", "std"],
            "y": [
                "min",
                "max",
            ],
        },
        depth="dataset",
    )
    check.is_instance(res, pd.DataFrame)


def test_dataset_agg_long_point(testset: Dataset):
    res = testset.agg(
        {
            "x": ["min", "max", "std"],
            "y": [
                "min",
                "max",
            ],
        },
        depth="point",
    )
    check.is_instance(res, pd.DataFrame)
    check.equal(48124, len(res))


def test_dataset_agg_long_pointcloud(testset: Dataset):
    res = testset.agg(
        {
            "x": ["min", "max", "std"],
            "y": [
                "min",
                "max",
            ],
        },
        depth="pointcloud",
    )
    check.is_instance(res, list)
    check.is_instance(res[0], pd.DataFrame)
    check.equal(2, len(res))
