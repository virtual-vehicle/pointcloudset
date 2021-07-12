import datetime
from pathlib import Path

import pandas as pd
import pytest
import pytest_check as check
from dask.dataframe import DataFrame as ddf

from pointcloudset import Dataset, PointCloud


def test_duration_type(testdataset_mini_real: Dataset):
    isinstance(type(testdataset_mini_real), datetime.timedelta)


def test_agg_wrong(testdataset_mini_real: Dataset):
    with pytest.raises(ValueError):
        testdataset_mini_real._agg("nix")


def test_agg_min(testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud):
    res = testdataset_mini_real._agg("min")
    computed = res.compute()
    check.is_instance(res, ddf)
    check.is_instance(computed, pd.DataFrame)
    check.equal(len(computed), len(testpointcloud_mini_real))
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


def test_agg_list(testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud):
    res = testdataset_mini_real._agg(["min", "max", "mean", "std"])
    computed = res.compute()
    check.is_instance(res, ddf)
    check.is_instance(computed, pd.DataFrame)
    check.is_instance(computed.columns, pd.core.indexes.multi.MultiIndex)
    check.equal(len(computed), len(testpointcloud_mini_real))


def test_agg_dict(testdataset_mini_real: Dataset, testpointcloud_mini_real: PointCloud):
    res = testdataset_mini_real._agg({"x": ["min", "max", "mean", "std"]})
    computed = res.compute()
    check.is_instance(res, ddf)
    check.is_instance(computed, pd.DataFrame)
    check.is_instance(computed.columns, pd.core.indexes.multi.MultiIndex)
    check.equal(len(computed), len(testpointcloud_mini_real))
