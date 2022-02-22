from pathlib import Path

import pytest
import pytest_check as check
from pointcloudset import Dataset
from pointcloudset.io.dataset.commandline import app
from typer.testing import CliRunner

from pyntcloud.io import TO_FILE

TO_FILE_PYNTCLOUD = list(TO_FILE.keys())

runner = CliRunner()


@pytest.fixture
def base_path() -> Path:
    """Get the current folder of the test"""
    return Path(__file__).parent


def test_something(base_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(base_path / "data")


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
    check.equal(result.exit_code, 0)
    check.equal(out_path.exists(), True)
    read_dataset = Dataset.from_file(out_path)
    check.is_instance(read_dataset, Dataset)
    check.equal(len(read_dataset), 2)
    check.equal(len(read_dataset.timestamps), 2)


def test_convert_all_bags_dir(
    tmp_path: Path, testdata_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(testdata_path)
    out_path = tmp_path.joinpath("cli_multi_dir")
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
    dirs = list(out_path.glob("**/*"))
    check.equal(len(dirs), 2)


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
    check.equal(out_path.exists(), True)
    files = list(out_path.glob(f"*.{fileformat.lower()}"))
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
    check.equal(result.exit_code, 0)
    check.equal(out_path.exists(), True)
    files = list(out_path.glob(f"*.{fileformat.lower()}"))
    check.equal(len(files), 1)
    check.equal(files[0].suffix.replace(".", ""), fileformat.lower())
