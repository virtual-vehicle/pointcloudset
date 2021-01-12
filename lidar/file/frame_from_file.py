from pathlib import Path

import pyntcloud

from ..frame import Frame


def frame_from_file(file_path: Path, **kwargs) -> Frame:
    """Generate a frame from a file. Any filetype supported by pyntcloud is also
    supported here.

    Args:
        file_path (Path): Path object to the file location

    Returns:
        Frame: frame object from the file
    """
    pyntcloud_in = pyntcloud.PyntCloud.from_file(file_path.as_posix(), **kwargs)
    frame = from_pyntcloud(pyntcloud_in)
    frame.orig_file = file_path.as_posix()
    return frame


def from_pyntcloud(pyntcloud_in: pyntcloud.PyntCloud) -> Frame:
    """Converts a pyntcloud to a lidar Frame.

    Args:
        pyntcloud_in (pyntcloud.PyntCloud): pyntcloud object to convert to frame

    Returns:
        Frame: Frame object from pyntcloud
    """
    data = pyntcloud_in.points
    return Frame(data=data)
