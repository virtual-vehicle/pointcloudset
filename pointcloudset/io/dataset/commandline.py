from pathlib import Path
from typing import Optional

import click  # needed for documentation
import typer

from pointcloudset import Dataset
from pointcloudset.io.dataset.bag import dataset_from_rosbag

app = typer.Typer()


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
    topic: str = typer.Argument("/os1_cloud_node/points"),
    folder_to_write: str = typer.Argument("."),
    start_frame_number: int = typer.Option(0, "--start", "-s"),
    end_frame_number: Optional[int] = typer.Option(None, "--end", "-e"),
    keep_zeros: bool = False,
    max_size: int = 100,
):
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
    typer.echo("done")


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
    typer_click_object()
