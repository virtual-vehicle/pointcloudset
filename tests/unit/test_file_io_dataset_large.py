# test on large fieles
# testfiles are not int the github repo since they are too large

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_check as check
from pandas._testing import assert_frame_equal
from pointcloudset import Dataset, PointCloud
from pointcloudset.io.dataset import dir


@pytest.mark.parametrize("filename", ["big_comp.bag", "big_uncomp.bag"])
def test_read_bag_big(testdata_path_large: Path, filename):
    if testdata_path_large.exists():
        dataset = Dataset.from_file(
            testdata_path_large.joinpath(filename), topic="/os1_cloud_node/points"
        )
        check.is_instance(dataset, Dataset)
        check.equal(len(dataset), 250)
