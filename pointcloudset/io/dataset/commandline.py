from pathlib import Path
from typing import Optional, Literal

import click  # needed for documentation
import typer

from pointcloudset import Dataset
from pointcloudset.io.dataset.bag import dataset_from_rosbag

from pyntcloud.io import TO_FILE

app = typer.Typer()

TO_FILE_PYNTCLOUD = list(TO_FILE.keys())
TO_FILE_CLI = TO_FILE_PYNTCLOUD.append("POINTCLOUDSET")


def _in_loop_for_cli(res, data, timestamps, folder_to_write, meta, chunk_number):
    data = res["data"]
    timestamps = res["timestamps"]
    Dataset(data, timestamps, meta).to_file(
        folder_to_write.joinpath(f"{chunk_number}"), use_orig_filename=False
    )


def _convert_bag2dir(
    bagfile: Path,
    topic: str,
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
    max_size: int = 100,
    folder_to_write: Path = Path(),
    mode="cli",
    in_loop_function=_in_loop_for_cli,
):
    return dataset_from_rosbag(**locals())


@app.command()
def get(
    bagfile: str,
    topic: str = typer.Option("/os1_cloud_node/points", "--topic", "-t"),
    folder_to_write: str = typer.Option(".", "--output-dir", "-d"),
    start_frame_number: int = typer.Option(0, "--start", "-s"),
    end_frame_number: Optional[int] = typer.Option(None, "--end", "-e"),
    output_format: str = typer.Option("POINTCLOUDSET", "--output-format", "-o"),
    keep_zeros: bool = False,
    max_size: int = 100,
):
    """The main CLI function to convert ROS bagfiles to pointcloudset or files supported
    by pyntloud.
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
        if output_format == "POINTCLOUDSET":
            _convert_bag2dir(
                bagfile=bagfile_path,
                topic=topic,
                folder_to_write=folder_to_write_path,
                start_frame_number=start_frame_number,
                end_frame_number=end_frame_number,
                keep_zeros=keep_zeros,
                max_size=max_size,
                in_loop_function=_in_loop_for_cli,
            )
        elif output_format.upper() in TO_FILE_PYNTCLOUD:
            dataset = Dataset.from_file(
                file_path=bagfile_path, topic=topic, keep_zeros=False
            )

            pyntcloud = dataset[0].to_instance("PYNTCLOUD")
            filename = folder_to_write_path.joinpath(
                "converted_1" + "." + output_format.lower()
            )
            typer.echo(f"writing file {filename} ")
            pyntcloud.to_file(filename.as_posix())

        else:
            raise typer.BadParameter(f"only one of {TO_FILE_CLI} is allowed")
    typer.echo("done")


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
    typer_click_object()
