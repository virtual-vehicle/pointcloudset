from pathlib import Path

import pandas as pd

import pytest_check as check
import pytest

from pointcloudset import Dataset, Frame
from dask.dataframe import DataFrame as ddf


def test_agg_wrong(testdataset_mini_real: Dataset):
    with pytest.raises(ValueError):
        testdataset_mini_real._agg("nix")


def test_agg_min(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    res = testdataset_mini_real._agg("min")
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
    res = testdataset_mini_real._agg(["min", "max", "mean", "std"])
    computed = res.compute()
    check.is_instance(res, ddf)
    check.is_instance(computed, pd.DataFrame)
    check.is_instance(computed.columns, pd.core.indexes.multi.MultiIndex)
    check.equal(len(computed), len(testframe_mini_real))


def test_agg_dict(testdataset_mini_real: Dataset, testframe_mini_real: Frame):
    res = testdataset_mini_real._agg({"x": ["min", "max", "mean", "std"]})
    computed = res.compute()
    check.is_instance(res, ddf)
    check.is_instance(computed, pd.DataFrame)
    check.is_instance(computed.columns, pd.core.indexes.multi.MultiIndex)
    check.equal(len(computed), len(testframe_mini_real))
