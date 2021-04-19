import datetime

import pytest
import pytest_check as check
import rosbag
from dask.delayed import DelayedLeaf
from pandas.testing import assert_series_equal
import pandas as pd

from lidar import Dataset, Frame


def test_dataset_len(testbag1: str, testset: Dataset):
    bag = rosbag.Bag(testbag1)
    len_bag = (bag.get_type_and_topic_info().topics)[
        "/os1_cloud_node/points"
    ].message_count
    check.equal(len_bag, len(testset))
    check.equal(len(testset), 2)


def test_getitem(testset: Dataset):
    check.equal(type(testset[0]), Frame)


def test_getitem_late(testset):
    check.equal(type(testset[1]), Frame)


def test_getitem_2times(testset):
    check.equal(type(testset[0]), Frame)
    check.equal(type(testset[1]), Frame)
    check.equal(type(testset[0]), Frame)
    check.equal(type(testset[1]), Frame)


def test_getitem_slice(testset: Dataset):
    test = testset[0:2]
    testlist = [0, 1, 2, 3]
    check.is_instance(test, Dataset)
    check.equal(len(test), 2)
    check.equal(len(test), len(testlist[0:2]))
    check.equal(type(testset[0:2][0]), Frame)


def test_getitem_error(testset: Dataset):
    with pytest.raises(TypeError):
        testset["fake"]


def test_get_frame_number_from_time(testset):
    res = testset._get_frame_number_from_time(testset.start_time)
    check.equal(res, 0)
    res2 = testset._get_frame_number_from_time(testset.end_time)
    check.equal(res2 + 1, len(testset))


def test_getitem_timerange(testset: Dataset):
    check.equal(len(testset), 2)
    dataset = testset.get_frames_between_timestamps(
        datetime.datetime(2020, 6, 22, 13, 40, 42, 657267),
        datetime.datetime(2020, 6, 22, 13, 40, 42, 755912),
    )
    check.equal(len(dataset), 2)
    check.equal(type(dataset[0:2]), Dataset)
    check.equal(type(dataset[0:2][0]), Frame)


def test_getitem_strange(testset):
    check.equal(len(testset), 2)
    check.equal(len(testset[2:0]), 0)


def test_has_frames(testset: Dataset):
    check.equal(testset.has_frames(), True)


def test_str(testset: Dataset):
    check.equal(type(str(testset)), str)
    check.equal(
        str(testset),
        "Lidar Dataset with 2 frame(s)",
    )


def test_repr(testset: Dataset):
    repr = testset.__repr__()
    check.equal(type(repr), str)
    check.equal(len(repr), 326)


def test_start_time(testset: Dataset):
    st = testset.start_time
    check.is_instance(st, datetime.datetime)
    check.equal(st, datetime.datetime(2020, 6, 22, 13, 40, 42, 657267))


def test_end_time(testset: Dataset):
    et = testset.end_time
    check.is_instance(et, datetime.datetime)
    check.equal(et, datetime.datetime(2020, 6, 22, 13, 40, 42, 755912))


def test_time(testset: Dataset):
    check.greater(testset.end_time, testset.start_time)


def test_extend(testbag1):
    ds1 = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    len_old = len(ds1)
    ds2 = Dataset.from_file(testbag1, topic="/os1_cloud_node/points", keep_zeros=True)
    ds1.extend(ds2)
    check.equal(len(ds1), len_old * 2)


def test_from_frames_list(testdataset_mini_real):
    check.is_instance(testdataset_mini_real, Dataset)
    check.equal(len(testdataset_mini_real), 2)
    check.is_instance(testdataset_mini_real.data[0], DelayedLeaf)
    check.is_instance(testdataset_mini_real[0], Frame)


def test_agg_frame(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    test = testdataset_mini_real._agg_per_frame("min")
    x_min = testframe_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(len(test), len(testdataset_mini_real))
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
            "timestamp",
        ],
    )
    check.equal(test.min()["x min"], x_min.values[0])


def test_agg_1(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    test = testdataset_mini_real.agg("min", "frame")
    x_min = testframe_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(len(test), len(testdataset_mini_real))
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
            "timestamp",
        ],
    )
    check.equal(test.min()["x min"], x_min.values[0])


def test_agg_dict1(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    test = testdataset_mini_real.agg({"x": "min"}, "frame")
    x_min = testframe_mini_real.data.agg({"x": "min"})
    check.is_instance(test, pd.DataFrame)
    check.equal(len(test), len(testdataset_mini_real))
    check.equal(
        list(test.columns),
        ["x {'x': 'min'}", "timestamp"],
    )
    check.equal(test.min()["x {'x': 'min'}"], x_min.values[0])


def test_agg_list1(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    f0 = testdataset_mini_real[0].data.agg(["min", "max"])
    f1 = testdataset_mini_real[1].data.agg(["min", "max"])
    test = testdataset_mini_real.agg(["min", "max"], "frame")
    check.is_instance(test, list)


def test_agg_dataset(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    test = testdataset_mini_real.agg("min", "dataset")
    x_min = testframe_mini_real.data.agg({"x": "min"})
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
        ],
    )
    check.equal(test["x min"], x_min.values[0])


def test_dataset_min1(
    testdataset_mini_real: Dataset,
    testframe_mini_real: Frame,
    testframe_mini_real_plus1: Frame,
):
    mincalc = testdataset_mini_real.min(depth=1)
    minshould = testframe_mini_real.data.drop("original_id", axis=1).min()
    check.is_instance(mincalc, pd.Series)
    assert_series_equal(mincalc, minshould, check_names=False)


def test_dataset_min0(
    testdataset_mini_real: Dataset,
    testframe_mini_real: Frame,
    testframe_mini_real_plus1: Frame,
):
    mincalc = testdataset_mini_real.min(depth=0)
    check.equal(mincalc.iloc[0].N, 2)
    check.is_instance(mincalc, pd.DataFrame)
    should = testframe_mini_real.extract_point(6008, use_orginal_id=True).squeeze()
    assert_series_equal(mincalc.drop("N", axis=1).iloc[0], should, check_names=False)
