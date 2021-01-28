from pathlib import Path

import pyntcloud

from ..frame import Frame
from ..convert import convert


def frame_from_file(file_path: Path, **kwargs) -> Frame:
    """Generate a frame from a file. Any filetype supported by pyntcloud is also
    supported here.

    Args:
        file_path (Path): Path object to the file location

    Returns:
        Frame: frame object from the file
    """
    pyntcloud_in = pyntcloud.PyntCloud.from_file(file_path.as_posix(), **kwargs)
    frame = convert.convert_pyntcloud2frame(pyntcloud_in)
    frame.orig_file = file_path.as_posix()
    return frame
