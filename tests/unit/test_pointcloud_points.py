import pandas as pd
import pytest
import pytest_check as check

from pointcloudset import PointCloud


@pytest.mark.parametrize("id, use_original_id", [(1, False), (4692, True)])
def test_extract_point(testpointcloud: PointCloud, id, use_original_id):
    res = testpointcloud.extract_point(id=id, use_original_id=use_original_id)
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


@pytest.mark.parametrize("use_original_id", [True, False])
def test_extract_point_not_available(testpointcloud: PointCloud, use_original_id):
    with pytest.raises(IndexError):
        testpointcloud.extract_point(id=10000000, use_original_id=use_original_id)
