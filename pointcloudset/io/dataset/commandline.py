from pathlib import Path
from typing import Optional

import click  # needed for documentation
import typer
from rich.console import Console


from pointcloudset import Dataset
from pointcloudset.io.dataset.bag import dataset_from_rosbag
import pointcloudset

from pyntcloud.io import TO_FILE

app = typer.Typer()
console = Console()

TO_FILE_PYNTCLOUD = list(TO_FILE.keys())
TO_FILE_CLI = TO_FILE_PYNTCLOUD.append("POINTCLOUDSET")


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
    console.line()
    console.rule(f"rosbagconvert  {pointcloudset.__version__}")
    bagfile_paths = _gen_bagfile_paths(bagfile)
    with console.status("Converting...", spinner="runner"):
        for bagfile_path in bagfile_paths:
            console.rule(f"converting {bagfile_path.name} ...", style="blue")

            folder_to_write_path = _gen_folder(folder_to_write, bagfile_path)

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


def _gen_bagfile_paths(bagfile):
    if bagfile == ".":
        bagfile_paths = list(Path.cwd().rglob("*.bag"))
    else:
        bagfile_paths = [Path(bagfile)]
    return bagfile_paths


def _gen_folder(folder_to_write, bagfile_path):
    if folder_to_write == ".":
        folder_to_write_path = Path.cwd().joinpath(bagfile_path.stem)
    else:
        folder_to_write_path = Path(folder_to_write).joinpath(bagfile_path.stem)

    if not folder_to_write_path.exists():
        folder_to_write_path.mkdir(parents=True, exist_ok=False)
    return folder_to_write_path


def _convert_bag2files(
    topic,
    start_frame_number,
    end_frame_number,
    output_format,
    bagfile_path,
    folder_to_write_path,
):
    """Converting a bagfile to files for each frame. Using pyntcloud

    Args:
        topic (_type_): _description_
        start_frame_number (_type_): _description_
        end_frame_number (_type_): _description_
        output_format (_type_): _description_
        bagfile_path (_type_): _description_
        folder_to_write_path (_type_): _description_
    """
    dataset = Dataset.from_file(
        file_path=bagfile_path,
        topic=topic,
        keep_zeros=False,
        start_frame_number=start_frame_number,
        end_frame_number=end_frame_number,
    )
    if end_frame_number is None:
        end_frame_number = len(dataset)
    for frame in range(start_frame_number, end_frame_number):
        pyntcloud = dataset[frame].to_instance("PYNTCLOUD")
        orig_file = Path(bagfile_path).stem
        filename = folder_to_write_path.joinpath(
            f"{orig_file}_{frame}.{output_format.lower()}"
        )
        console.print(
            f"frame {frame} of {Path(bagfile_path).name} converted to {filename}"
        )
        pyntcloud.to_file(filename.as_posix())


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
    typer_click_object()
