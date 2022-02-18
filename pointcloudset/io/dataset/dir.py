import json
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
DELIMITER = ";"


def dataset_to_dir(
    dataset_in, file_path: Path, use_orig_filename: bool = True, **kwargs
) -> None:
    """Writes Dataset to directory.

    Args:
        dataset_in (Dataset): Dataset to write.
        file_path (pathlib.Path): Destination path.
        use_orig_filename (bool): Use filename from which the dataset was read. Defaults to ``True``.
        **kwargs: Keyword arguments to pass to dask to_parquet function
    """
    if not dataset_in.has_pointclouds():
        raise ValueError("dataset must have data ")
    _check_dir(file_path)
    orig_filename = Path(dataset_in.meta["orig_file"]).stem
    if len(orig_filename) == 0:
        orig_filename = str(uuid.uuid4())
    folder = file_path.joinpath(orig_filename) if use_orig_filename else file_path
    empty_data = _get_empty_data(dataset_in)
    dataset_to_write = dataset_in._replace_empty_frames_with_nan(empty_data)
    data = dd.from_delayed(dataset_to_write.data)
    data.to_parquet(folder, **kwargs)
    meta = dataset_in.meta
    meta["timestamps"] = [
        timestamp.strftime(DATETIME_FORMAT) for timestamp in dataset_in.timestamps
    ]
    meta["empty_data"] = empty_data.to_dict()
    with open(folder.joinpath("meta.json"), "w") as outfile:
        json.dump(dataset_in.meta, outfile)
    _check_dir_contents(folder)
    print(f"Files written to: {folder}")


def dataset_from_dir(dir: Path) -> dict:
    # sourcery skip: simplify-len-comparison
    """Reads a Dataset from a directory.

    Args:
        dir (pathlib.Path): Path of directory.

    Returns:
        dict: Lidar data with timestamps and metadata.
    """
    _check_dir(dir)
    dirs = [e for e in dir.iterdir() if e.is_dir()]

    if len(dirs) > 0:
        dirs.sort(key=_get_folder_number)
    else:
        dirs = [dir]

    data = []
    timestamps = []
    meta = []
    for path in dirs:
        res = _dataset_from_single_dir(path)
        data.extend(res["data"])
        timestamps.extend(res["timestamps"])
        meta.append(res["meta"])
    meta = meta[0]
    del meta["timestamps"]
    if "empty_data" in meta:
        empty_data = pd.DataFrame.from_dict(meta["empty_data"])
    else:
        empty_data = pd.DataFrame()
        # for backwards compatibility
    return {
        "data": data,
        "timestamps": timestamps,
        "meta": meta,
        "empty_data": empty_data,
    }


def _get_folder_number(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        print(path)
        raise ValueError("Not a path with a dataset.")


def _dataset_from_single_dir(dir: Path) -> dict:
    _check_dir(dir)
    data = dd.read_parquet(dir)
    with open(dir.joinpath("meta.json"), "r") as infile:
        meta = json.loads(infile.read())
    timestamps_raw = meta["timestamps"]
    timestamps = [
        datetime.strptime(timestamp, DATETIME_FORMAT) for timestamp in timestamps_raw
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


def _check_dir_contents(dir: Path):
    """Quick test if all"""
    sub_dirs = list(dir.glob("**"))
    for path in sub_dirs:
        if len(sub_dirs) != 1 and path != dir or len(sub_dirs) == 1:
            _check_dir_contents_single(path)


def _check_dir_contents_single(dir: Path):
    """checking the folder content of a written dataset."""
    assert dir.joinpath("meta.json").is_file(), f"meta.json is missing in {dir}"
    assert dir.joinpath("part.0.parquet"), f"part.0.parquet is missing in {dir}"


def _get_empty_data(dataset_in) -> pd.DataFrame:
    """Get an row of data to use as placeholder for empty datasets.

    Args:
        dataset_in (Dataset): Dataset which contains at least on valid pointcloud.

    Returns:
        pd.DataFrame: empta data placeholder
    """
    empty_data = pd.DataFrame.from_dict(
        {
            "x": [np.nan],
            "y": [np.nan],
            "z": [np.nan],
            "original_id": [np.nan],
        }
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for pointcloud in dataset_in:
            if len(pointcloud) > 0:
                empty_data = pd.DataFrame(pointcloud.data.iloc[0]).T
                break
    return empty_data
