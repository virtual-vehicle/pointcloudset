from pathlib import Path

import pytest_check as check
from typer.testing import CliRunner

from pointcloudset import Dataset
from pointcloudset.io.dataset.commandline import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    check.equal(result.exit_code, 0)
    check.equal("Usage:" in result.stdout, True)


def test_convert(testbag1: Path, tmp_path: Path):
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
