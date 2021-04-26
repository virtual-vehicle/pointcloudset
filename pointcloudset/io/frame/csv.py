from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pointcloudset import Frame


def write_csv(frame: Frame, file_path: Path) -> None:
    """Exports the frame as a csv for use with cloud compare or similar tools.
    Currently not all attributes of a frame are saved so some information is lost when
    using this function.

    Args:
        file_path (Path): Destination.
    """
    frame.data.to_csv(file_path, index=False)