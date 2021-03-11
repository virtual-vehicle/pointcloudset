from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from lidar import Frame


def write_csv(frame: Frame, file_path: Path = Path()) -> None:
    """Exports the frame as a csv for use with cloud compare or similar tools.
    Currently not all attributes of a frame are saved so some information is lost when
    using this function.

    Args:
        file_path (Path, optional): Destination. Defaults to the folder of
        the bag fiile with the timestamp of the frame.
    """
    orig_file_name = Path(frame.orig_file).stem
    if file_path == Path():
        filename = f"{orig_file_name}_timestamp_{frame.timestamp}.csv"
        destination_folder = Path(frame.orig_file).parent.joinpath(filename)
    else:
        destination_folder = file_path
    frame.data.to_csv(destination_folder, index=False)