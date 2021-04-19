from pathlib import Path

import pandas as pd
from pandas.testing import assert_series_equal
import pytest_check as check
import pytest

from lidar import Dataset, Frame
from lidar.dataset_core import DatasetCore
from dask.dataframe import DataFrame as ddf


def test_agg_wrong(testdataset_mini_real: Dataset):
    with pytest.raises(AssertionError):
        testdataset_mini_real.agg("nix")


def test_agg_min(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    res = testdataset_mini_real.agg("min")
    computed = res.compute()
    check.is_instance(res, ddf)
    check.is_instance(computed, pd.DataFrame)
    check.equal(len(computed), len(testframe_mini_real))
    check.equal(
        list(computed.columns),
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
            "N",
            "original_id",
        ],
    )


def test_agg_list(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    res = testdataset_mini_real.agg(["min", "max", "mean", "std"])
    computed = res.compute()
    check.is_instance(res, ddf)
    check.is_instance(computed, pd.DataFrame)
    check.is_instance(computed.columns, pd.core.indexes.multi.MultiIndex)
    check.equal(len(computed), len(testframe_mini_real))


def test_agg_dict(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    res = testdataset_mini_real.agg({"x": ["min", "max", "mean", "std"]})
    computed = res.compute()
    check.is_instance(res, ddf)
    check.is_instance(computed, pd.DataFrame)
    check.is_instance(computed.columns, pd.core.indexes.multi.MultiIndex)
    check.equal(len(computed), len(testframe_mini_real))


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


def test_dataset_minframe(
    testdataset_mini_real: Dataset,
    testframe_mini_real: Frame,
    testframe_mini_real_plus1: Frame,
):
    min_x_frame = testdataset_mini_real.min("frame", "x")
    min_x_point = testdataset_mini_real.min("point", "x")
    min_x_dataset = testdataset_mini_real.min("dataset", "x")
    min_all = testdataset_mini_real.min("frame")
    min_x_y_z = testdataset_mini_real.min(depth="dataset", columns=["x", "y", "z"])
