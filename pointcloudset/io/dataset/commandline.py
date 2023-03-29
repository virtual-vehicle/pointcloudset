from __future__ import annotations
from pathlib import Path

import click  # needed for documentation
import pointcloudset
import typer
from pointcloudset import Dataset
from pyntcloud.io import TO_FILE
from rich.console import Console

from typing import Union

app = typer.Typer()
console = Console()

TO_FILE_PYNTCLOUD = list(TO_FILE.keys())
TO_FILE_CLI = TO_FILE_PYNTCLOUD.append("POINTCLOUDSET")


@app.command()
def get(
    ros_file: str,
    topic: str = typer.Option("/os1_cloud_node/points", "--topic", "-t"),
    folder_to_write: str = typer.Option(".", "--output-dir", "-d"),
    output_format: str = typer.Option("POINTCLOUDSET", "--output-format", "-o"),
    start_frame_number: int = typer.Option(0, "--start", "-s"),
    end_frame_number: Union[int, None] = typer.Option(None, "--end", "-e"),
    keep_zeros: bool = False,
):
    """The main CLI function to convert ROS1 and ROS2 files to pointcloudset or files
    supported by pyntloud.

    Examples:

    convert all ROS1 bag files in a directory
    $ pointcloudset-convert -d converted .

    convert all frames of bagfile xyz.bag into csv files
    $ pointcloudset-convert -o csv -d converted_csv xyz.bag

    convert a ROS2 directoy to a pointcloudset file
    $ pointcloudset-convert -d converted something_ros2

    convert the first 10 frames of a bag file int0las files
    $ pointcloudset-convert -o las -d converted_las --start 1 --end 10 xyz.bag
    """
    console.line()
    console.rule(f"pointcloudset-convert  {pointcloudset.__version__}")
    bagfile_paths = _gen_file_paths(ros_file)
    console.rule(output_format)
    with console.status("Converting...", spinner="runner"):
        for bagfile_path in bagfile_paths:
            console.rule(f"converting {bagfile_path.name} ...", style="blue")

            folder_to_write_path = _gen_folder(folder_to_write, bagfile_path)

            if output_format == "POINTCLOUDSET":
                _convert_one_bag2dir(
                    ros_file=bagfile_path,
                    topic=topic,
                    start_frame_number=start_frame_number,
                    end_frame_number=end_frame_number,
                    keep_zeros=keep_zeros,
                    folder_to_write=folder_to_write_path,
                )
                console.print(
                    f"{Path(bagfile_path).name} converted to {folder_to_write_path}"
                )
            elif output_format.upper() in TO_FILE_PYNTCLOUD:
                _convert_bag2files(
                    topic,
                    start_frame_number,
                    end_frame_number,
                    output_format,
                    bagfile_path,
                    folder_to_write_path,
                )

            else:
                raise typer.BadParameter(f"only one of {TO_FILE_CLI} is allowed")
    console.rule("Done :sake:")


def _convert_one_bag2dir(
    ros_file: Path,
    topic: str,
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
    folder_to_write: Path = Path(),
):
    if not ros_file.exists():
        raise typer.BadParameter(f"{ros_file} does not exist")
    dataset = Dataset.from_file(
        file_path=ros_file,
        topic=topic,
        start_frame_number=start_frame_number,
        end_frame_number=end_frame_number,
        keep_zeros=keep_zeros,
    )
    if len(dataset) > 0:
        dataset.to_file(
            file_path=folder_to_write,
            use_orig_filename=False,
        )
    else:
        console.print("no data, skipping")


def _gen_file_paths(file_name):
    if file_name == ".":
        bagfile_paths = list(Path.cwd().glob("*.bag"))
    else:
        bagfile_paths = [Path(file_name)]
    return bagfile_paths


def _gen_folder(folder_to_write: Path, ros_file_path: Path) -> Path:
    folder_to_write_path = Path(folder_to_write).joinpath(
        ros_file_path.stem + "_pointcloudset"
    )

    if not folder_to_write_path.exists():
        folder_to_write_path.mkdir(exist_ok=False, parents=True)

    return folder_to_write_path


def _convert_bag2files(
    topic,
    start_frame_number,
    end_frame_number,
    output_format,
    ros_file_path,
    folder_to_write_path,
):
    """Converting a bagfile to files for each frame. Using pyntcloud

    Args:
        topic (_type_): _description_
        start_frame_number (_type_): _description_
        end_frame_number (_type_): _description_
        output_format (_type_): _description_
        ros_file_path (_type_): _description_
        folder_to_write_path (_type_): _description_
    """
    dataset = Dataset.from_file(
        file_path=ros_file_path,
        topic=topic,
        keep_zeros=False,
        start_frame_number=start_frame_number,
        end_frame_number=end_frame_number,
    )
    if end_frame_number is None:
        end_frame_number = len(dataset)
    for frame in range(start_frame_number, end_frame_number):
        pyntcloud = dataset[frame].to_instance("PYNTCLOUD")
        orig_file = Path(ros_file_path).stem
        filename = folder_to_write_path.joinpath(
            f"{orig_file}_{frame}.{output_format.lower()}"
        )
        console.print(
            f"frame {frame} of {Path(ros_file_path).name} converted to {filename}"
        )
        pyntcloud.to_file(filename.as_posix())


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
    typer_click_object()
