from pathlib import Path

import pandas as pd
from pandas.testing import assert_series_equal
import pytest_check as check

from lidar import Dataset
from lidar.dataset_core import DatasetCore


def test_dataset_min(
    testdataset_mini_real: Dataset, testframe_mini_real, testframe_mini_real_plus1
):
    mincalc = testdataset_mini_real.min(depth=1)
    minshould = testframe_mini_real.data.drop("original_id", axis=1).min()
    check.is_instance(mincalc, pd.Series)
    assert_series_equal(mincalc, minshould, check_names=False)
