from pathlib import Path
from typing import Optional

import numpy as np
import rosbag
import typer

import lidar
from lidar.io.dataset.bag import get_number_of_messages, read_rosbag_part

app = typer.Typer()


def convert_bag2dir(
    bagfile: Path,
    folder_to_write: Path,
    topic: str,
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
    max_size: int = 100,
) -> dict:
    bag = rosbag.Bag(bagfile.as_posix())
    max_messages = get_number_of_messages(bag, topic)

    if end_frame_number is None:
        end_frame_number = max_messages
    if end_frame_number > max_messages:
        raise ValueError("end_frame_number to high")

    framelist = np.arange(start_frame_number, end_frame_number)

    chunks = np.array_split(framelist, int(np.ceil(len(framelist) / max_size)))
    data = []
    timestamps = []
    meta = {"orig_file": bagfile.as_posix(), "topic": topic}
    chunk_number = 0
    for chunk in chunks:
        res = read_rosbag_part(
            bag,
            topic=topic,
            start_frame_number=chunk[0],
            end_frame_number=chunk[-1] + 1,
            keep_zeros=keep_zeros,
        )
        data = res["data"]
        timestamps = res["timestamps"]
        lidar.Dataset(data, timestamps, meta).to_file(
            folder_to_write.joinpath(f"{chunk_number}"), use_orig_filename=False
        )
        chunk_number = chunk_number + 1


@app.command()
def get(
    bagfile: str,
    folder_to_write: str = typer.Argument("."),
    topic: str = typer.Argument("/os1_cloud_node/points"),
    start_frame_number: int = typer.Option(0, "--start", "-s"),
    end_frame_number: Optional[int] = typer.Option(None, "--end", "-e"),
    keep_zeros: bool = False,
    max_size: int = 100,
):
    """Convert ROS bagfiles to a directory of use with the lidar package.

    Args:
        bagfile: ROS bagfile
        folder_to_write: [description]
        topic: ros lidar pointcloud topic. For example "/os1_cloud_node/points".
        start_frame_number:  Defaults to 0.
        end_frame_number:  Defaults to None.
        keep_zeros: Keep element with zero values. Defaults to ``False``.
        max_size: Max size of chunk, an internal variable. Defaults to 100.
    """

    if bagfile == ".":
        bagfile_paths = list(Path.cwd().rglob("*.bag"))
    else:
        bagfile_paths = [Path(bagfile)]
    for bagfile_path in bagfile_paths:
        typer.echo(f"converting {bagfile_path.name} ...")
        if folder_to_write == ".":
            folder_to_write_path = Path.cwd().joinpath(bagfile_path.stem)
        else:
            folder_to_write_path = Path(folder_to_write)
        convert_bag2dir(
            bagfile_path,
            folder_to_write_path,
            topic,
            start_frame_number,
            end_frame_number,
            keep_zeros,
            max_size,
        )
    typer.echo("done")


if __name__ == "__main__":
    app()
