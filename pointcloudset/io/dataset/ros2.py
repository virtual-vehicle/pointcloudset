from pathlib import Path


def dataset_from_ros2(
    bagfile: Path,
    topic: str,
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
):
    raise NotImplementedError("ROS2 files are currently not supported")
