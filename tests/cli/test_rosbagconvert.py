from pathlib import Path

import pytest
import pytest_check as check
from pointcloudset import Dataset
from pointcloudset.io.dataset.commandline import app
from typer.testing import CliRunner

from pyntcloud.io import TO_FILE

TO_FILE_PYNTCLOUD = list(TO_FILE.keys())

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    check.equal(result.exit_code, 0)
    check.equal("Usage:" in result.stdout, True)


def test_convert_one_bag_dir(testbag1: Path, tmp_path: Path):
    out_path = tmp_path.joinpath("cli")
    result = runner.invoke(
        app,
        [
            testbag1.as_posix(),
            "-t",
            "/os1_cloud_node/points",
            "-d",
            out_path.as_posix(),
        ],
    )
    out_path_real = out_path.joinpath("test")
    check.equal(result.exit_code, 0)
    check.equal(out_path_real.exists(), True)
    read_dataset = Dataset.from_file(out_path_real)
    check.is_instance(read_dataset, Dataset)
    check.equal(len(read_dataset), 2)
    check.equal(len(read_dataset.timestamps), 2)


def test_convert_all_bags_dir(
    tmp_path: Path, testdata_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(testdata_path)
    out_path = tmp_path.joinpath("cli_dirs")
    result = runner.invoke(
        app,
        [
            ".",
            "-t",
            "/os1_cloud_node/points",
            "-d",
            out_path.as_posix(),
        ],
    )
    check.equal(result.exit_code, 0)
    check.equal(out_path.exists(), True)
    dirs = [f for f in out_path.iterdir() if f.is_dir()]
    check.equal(len(dirs), 2)
    read_dataset = Dataset.from_file(dirs[0])
    check.is_instance(read_dataset, Dataset)
    check.equal(len(read_dataset), 2)
    check.equal(len(read_dataset.timestamps), 2)
    read_dataset = Dataset.from_file(dirs[1])
    check.is_instance(read_dataset, Dataset)
    check.equal(len(read_dataset), 2)
    check.equal(len(read_dataset.timestamps), 2)


@pytest.mark.parametrize("fileformat", TO_FILE_PYNTCLOUD)
def test_convert_one_bag_frames_to_files(testbag1: Path, tmp_path: Path, fileformat):
    out_path = tmp_path.joinpath("cli_files")
    result = runner.invoke(
        app,
        [
            testbag1.as_posix(),
            "-t",
            "/os1_cloud_node/points",
            "-d",
            out_path.as_posix(),
            "-o",
            fileformat.lower(),
        ],
    )
    check.equal(result.exit_code, 0)
    out_path_real = out_path.joinpath("test")
    check.equal(out_path_real.exists(), True)
    files = list(out_path_real.glob(f"*.{fileformat.lower()}"))
    check.equal(len(files), 2)
    check.equal(files[0].suffix.replace(".", ""), fileformat.lower())


@pytest.mark.parametrize("fileformat", TO_FILE_PYNTCLOUD)
def test_convert_one_bag_one_frames_to_file(testbag1: Path, tmp_path: Path, fileformat):
    out_path = tmp_path.joinpath("cli_1file")
    result = runner.invoke(
        app,
        [
            testbag1.as_posix(),
            "-t",
            "/os1_cloud_node/points",
            "-d",
            out_path.as_posix(),
            "-o",
            fileformat.lower(),
            "-s",
            "0",
            "-e",
            "1",
        ],
    )
    out_path_real = out_path.joinpath("test")
    check.equal(result.exit_code, 0)
    check.equal(out_path_real.exists(), True)
    files = list(out_path_real.glob(f"*.{fileformat.lower()}"))
    check.equal(len(files), 1)
    check.equal(files[0].suffix.replace(".", ""), fileformat.lower())


def test_convert_all_bags_frames_files(
    tmp_path: Path, testdata_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(testdata_path)
    out_path = tmp_path.joinpath("cli_dirs_frames")
    result = runner.invoke(
        app,
        [".", "-t", "/os1_cloud_node/points", "-d", out_path.as_posix(), "-o", "csv"],
    )
    check.equal(result.exit_code, 0)
    check.equal(out_path.exists(), True)
    dirs = [f for f in out_path.iterdir() if f.is_dir()]
    check.equal(len(dirs), 2)
    files = list(dirs[0].glob("*.csv"))
    check.equal(len(files), 2)
    check.equal(files[0].suffix.replace(".", ""), "csv")


@pytest.mark.slow
@pytest.mark.parametrize("filename", ["big_uncomp.bag", "big_comp.bag"])
def test_convert_large_file_complete(
    testdata_path_large: Path, tmp_path: Path, filename: str
):
    if testdata_path_large.exists():
        len_target = 250
        testbag = testdata_path_large.joinpath(filename)
        check.is_true(testbag.exists())
        out_path = tmp_path.joinpath("cli")
        result = runner.invoke(
            app,
            [
                testbag.as_posix(),
                "-t",
                "/os1_cloud_node/points",
                "-d",
                out_path.as_posix(),
            ],
        )
        out_path_real = out_path / Path(filename).stem
        check.equal(result.exit_code, 0)
        check.equal(out_path_real.exists(), True)
        check.equal(len(list(out_path_real.parent.glob("*/*"))), len_target + 1)
        read_dataset = Dataset.from_file(out_path_real)
        check.is_instance(read_dataset, Dataset)
        check.equal(len(read_dataset), len_target)
        check.equal(len(read_dataset.timestamps), len_target)


@pytest.mark.slow
def test_convert_large_file_part1(testdata_path_large: Path, tmp_path: Path):
    if testdata_path_large.exists():
        filename = "big_uncomp.bag"
        len_target = 30
        testbag = testdata_path_large.joinpath(filename)
        check.is_true(testbag.exists())
        out_path = tmp_path.joinpath("cli")
        result = runner.invoke(
            app,
            [
                testbag.as_posix(),
                "-s",
                "50",
                "-e",
                "80",
                "-t",
                "/os1_cloud_node/points",
                "-d",
                out_path.as_posix(),
            ],
        )
        out_path_real = out_path / Path(filename).stem
        check.equal(result.exit_code, 0)
        check.equal(len(list(out_path_real.parent.glob("*/*"))), len_target + 1)
        check.equal(out_path_real.exists(), True)
        read_dataset = Dataset.from_file(out_path_real)
        check.is_instance(read_dataset, Dataset)
        check.equal(len(read_dataset), len_target)
        check.equal(len(read_dataset.timestamps), len_target)
