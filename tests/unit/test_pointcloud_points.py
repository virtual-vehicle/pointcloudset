import numpy as np
import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import Dataset, PointCloud


def test_extract_point1(testpointcloud: PointCloud):
    res = testpointcloud.extract_point(id=1, use_original_id=False)
    check.equal(type(res), pd.DataFrame)
    check.equal(len(res), 1)
    check.equal(list(res.index.values)[0], 0)
    types = [str(types) for types in res.dtypes.values]
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


def test_extract_point_not_available(testpointcloud: PointCloud):
    with pytest.raises(IndexError):
        testpointcloud.extract_point(id=10000000, use_original_id=False)


def test_extract_point_orginal_id(testpointcloud: PointCloud):
    res = testpointcloud.extract_point(id=4692, use_original_id=True)
    check.equal(type(res), pd.DataFrame)
    check.equal(len(res), 1)
    check.equal(list(res.index.values)[0], 0)
    types = [str(types) for types in res.dtypes.values]
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


def test_extract_point_orginal_id_not_available(testpointcloud: PointCloud):
    with pytest.raises(IndexError):
        testpointcloud.extract_point(id=10000000, use_original_id=True)
