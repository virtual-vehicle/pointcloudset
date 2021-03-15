from typing import TYPE_CHECKING
from pathlib import Path
import dask.dataframe as dd
import json
import numpy as np
import uuid


def dataset_to_dir(dataset_in, file_path: Path) -> None:
    if not isinstance(file_path, Path):
        raise TypeError("expecting a pathlib Path object")
    if len(file_path.suffix) != 0:
        raise ValueError("expecting a path not a filename")
    orig_filename = Path(dataset_in.meta["orig_file"]).stem
    if len(orig_filename) == 0:
        orig_filename = str(uuid.uuid4())
    folder = file_path.joinpath(orig_filename)
    data = dd.from_delayed(dataset_in.data)
    data.to_parquet(folder)
    with open(folder.joinpath("meta.json"), "w") as outfile:
        json.dump(dataset_in.meta, outfile)
    np.savetxt(
        folder.joinpath("timestamps.txt"),
        dataset_in.timestamps,
        delimiter=" ",
        fmt="%s",
    )
    print(f"Files written to: {folder}")
