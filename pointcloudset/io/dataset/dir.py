import json
import uuid
from datetime import datetime
from pathlib import Path

import dask.dataframe as dd

datetime_format = "%Y-%m-%d %H:%M:%S.%f"
delimiter = ";"


def dataset_to_dir(dataset_in, file_path: Path, use_orig_filename: bool = True) -> None:
    """Writes Dataset to directory.

    Args:
        dataset_in (Dataset): Dataset to write.
        file_path (pathlib.Path): Destination path.
        use_orig_filename (bool): Use filename from which the dataset was read. Defaults to ``True``.
    """
    _check_dir(file_path)
    orig_filename = Path(dataset_in.meta["orig_file"]).stem
    if len(orig_filename) == 0:
        orig_filename = str(uuid.uuid4())
    folder = file_path.joinpath(orig_filename) if use_orig_filename else file_path
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
    return {
        "data": data,
        "timestamps": timestamps,
        "meta": meta,
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
