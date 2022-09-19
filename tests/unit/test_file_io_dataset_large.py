# test on large files
# testfiles are not int the github repo since they are too large
from pathlib import Path

import pytest
import pytest_check as check
from pointcloudset import Dataset


@pytest.mark.parametrize("filename", ["big_comp.bag", "big_uncomp.bag", "big_1000.bag"])
def test_read_and_write_bag_big(testdata_path_large: Path, filename, tmp_path: Path):
    if testdata_path_large.exists():
        # read
        ds = Dataset.from_file(
            testdata_path_large.joinpath(filename),
            topic="/os1_cloud_node/points",
            keep_zeros=True,
        )
        check.is_instance(ds, Dataset)
        check.equal(len(ds), 250)
        testfile_name = tmp_path.joinpath("dataset")
        # write
        ds.to_file(file_path=testfile_name, use_orig_filename=False)
        p = testfile_name.glob("*.parquet")
        files = [x for x in p if x.is_file()]
        check.equal(len(files), 250)
        meta_gen = testfile_name.glob("meta.json")
        metafile = list(meta_gen)[0]
        check.equal(metafile.exists(), True)
        check.equal(testfile_name.exists(), True)
        # read again
        read_dataset = Dataset.from_file(testfile_name)
        check.is_instance(read_dataset, Dataset)
        check.equal(len(ds), len(read_dataset))
