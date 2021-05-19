from typing import Any

import pytest
import pytest_check as check

import pointcloudset as pcs
from pointcloudset.pipeline.delayed_result import DelayedResult


def first(pc: pcs.PointCloud) -> Any:
    return pc.data.iloc[0]


@pytest.fixture
def res(testdataset_mini_real):
    return testdataset_mini_real.apply(first, warn=False)


def test_delayed_result(res):
    res_computed = res.compute()
    check.is_instance(res, DelayedResult)
    check.equal(res_computed[0]["x"], 0.9805122017860413)


def test_delayed_result0(res):
    res0 = res[0]
    check.equal(res0["x"], 0.9805122017860413)


def test_delayed_result_wrong(res):
    with pytest.raises(TypeError):
        res0 = res["nix"]


def test_delayed_result_slice(res):
    res02 = res[0:2]
    check.is_instance(res02, list)
    check.equal(res02[0]["x"], 0.9805122017860413)
