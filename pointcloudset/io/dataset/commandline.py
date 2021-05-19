from pathlib import Path
from typing import Optional

import typer
import click  # needed for documentation

from pointcloudset.io.dataset.convert_bag2dataset import convert_bag2dir

app = typer.Typer()


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


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
    typer_click_object()
