import lidar
from pyntcloud import PyntCloud
from pathlib import Path
import pytest_check as check


def test_pyntcloud(testlas1: Path):
    pyntcloud_data = PyntCloud.from_file(testlas1.as_posix())
    frame = lidar.Frame.from_instance("pyntcloud", pyntcloud_data)
    check.is_instance(frame, lidar.Frame)
    check.equal(frame.has_original_id(), False)