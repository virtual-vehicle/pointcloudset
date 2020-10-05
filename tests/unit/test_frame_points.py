import pandas as pd
import numpy as np
import pytest
import pytest_check as check

from lidar import Frame
from lidar import Dataset


def test_extract_point1(testframe: Frame):
    res = testframe.extract_point(id=1, use_orginal_id=False)
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


def test_extract_point_not_available(testframe: Frame):
    with pytest.raises(IndexError):
        testframe.extract_point(id=10000000, use_orginal_id=False)


def test_extract_point_orginal_id(testframe: Frame):
    res = testframe.extract_point(id=4692, use_orginal_id=True)
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


def test_extract_point_orginal_id_not_available(testframe: Frame):
    with pytest.raises(IndexError):
        testframe.extract_point(id=10000000, use_orginal_id=True)


def test_point_difference(testframe: Frame):
    difference = testframe.point_difference(testframe, 4624)
    check.equal(len(difference), 1)
    check.equal(
        (
            difference.values
            == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4624.0]])
        ).all(),
        True,
    )
    check.equal(
        (
            (difference.columns).values
            == np.array(
                [
                    "x difference",
                    "y difference",
                    "z difference",
                    "intensity difference",
                    "t difference",
                    "reflectivity difference",
                    "ring difference",
                    "noise difference",
                    "range difference",
                    "original_id",
                ]
            )
        ).all(),
        True,
    )


def test_point_difference2(testset: Dataset):
    testframe1 = testset[0]
    testframe2 = testset[1]
    diff = testframe1.point_difference(testframe2, 4692)
    check.equal(len(diff), 1)
    check.equal(diff.original_id.values[0], 4692)
    check.equal(
        np.allclose(
            diff.values,
            [
                -5.73961735e-02,
                1.63454115e-02,
                -6.20609522e-03,
                -3.00000000e00,
                3.40000000e02,
                6.55350000e04,
                0.00000000e00,
                1.40000000e01,
                4.29496724e09,
                4.69200000e03,
            ],
        ),
        True,
    )
    types = [str(types) for types in diff.dtypes.values]
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
