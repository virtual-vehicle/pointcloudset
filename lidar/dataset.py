"""
# Dataset Class
The Dataset class which contains many frames.

For more details on how to use it please refer to the usage.ipynb Notebook for an interactive tuturial.

# Developer notes
* The important stuff happens in the __getitem__ method. Only then the rosbag is actually read with the help of
generators.
"""
from .dataset_core import DatasetCore

from pathlib import Path
from .io import DATASET_FROM_FILE


class Dataset(DatasetCore):
    @classmethod
    def from_file(cls, file_path: Path, **kwargs):
        if not isinstance(file_path, Path):
            raise TypeError("Expectinga Path object for file_path")
        ext = file_path.suffix[1:].upper()
        if ext not in DATASET_FROM_FILE:
            raise ValueError(
                "Unsupported file format; supported formats are: {}".format(
                    list(DATASET_FROM_FILE)
                )
            )
        else:
            res = DATASET_FROM_FILE[ext](file_path, **kwargs)
            return cls(data=res["data"], timestamps=res["timestamps"], meta=res["meta"])
