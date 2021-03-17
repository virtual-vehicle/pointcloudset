import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import dask.dataframe as dd
import numpy as np


datetime_format = "%Y-%m-%d %H:%M:%S.%f"
delimiter = ";"


def dataset_to_dir(dataset_in, file_path: Path, use_orig_filename: bool = True) -> None:
    _check_dir(file_path)
    orig_filename = Path(dataset_in.meta["orig_file"]).stem
    if len(orig_filename) == 0:
        orig_filename = str(uuid.uuid4())
    if use_orig_filename:
        folder = file_path.joinpath(orig_filename)
    else:
        folder = file_path
    data = dd.from_delayed(dataset_in.data)
    data.to_parquet(folder)
    meta = dataset_in.meta
    meta["timestamps"] = [
        timestamp.strftime(datetime_format) for timestamp in dataset_in.timestamps
    ]
    with open(folder.joinpath("meta.json"), "w") as outfile:
        json.dump(dataset_in.meta, outfile)
    print(f"Files written to: {folder}")


def dataset_from_dir(dir: Path) -> dict:
    _check_dir(dir)
    dirs = [e for e in dir.iterdir() if e.is_dir()]
    dirs.sort()

    if len(dirs) == 0:
        dirs = [dir]

    data = []
    timestamps = []
    meta = []
    for path in dirs:
        print(path)
        res = _dataset_from_single_dir(path)
        data.extend(res["data"])
        timestamps.extend(res["timestamps"])
        meta.append(res["meta"])
        print(len(timestamps))
    meta = meta[0]
    del meta["timestamps"]
    print(len(timestamps))
    return {
        "data": data,
        "timestamps": timestamps,
        "meta": meta,
    }


def _dataset_from_single_dir(dir: Path) -> dict:
    _check_dir(dir)
    data = dd.read_parquet(dir)
    with open(dir.joinpath("meta.json"), "r") as infile:
        meta = json.loads(infile.read())
    timestamps_raw = meta["timestamps"]
    timestamps = [
        datetime.strptime(timestamp, datetime_format) for timestamp in timestamps_raw
    ]
    return {
        "data": data.to_delayed(),
        "timestamps": timestamps,
        "meta": meta,
    }


def _check_dir(file_path: Path):
    if not isinstance(file_path, Path):
        raise TypeError("expecting a pathlib Path object")
    if len(file_path.suffix) != 0:
        raise ValueError("expecting a path not a filename")
