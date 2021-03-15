"""
# Dataset Class
The Dataset class which contains many frames.

For more details on how to use it please refer to the usage.ipynb Notebook for an interactive tuturial.

# Developer notes
* The important stuff happens in the __getitem__ method. Only then the rosbag is actually read with the help of
generators.
"""
from pathlib import Path

from .dataset_core import DatasetCore
from .io import DATASET_FROM_FILE, DATASET_TO_FILE

from typing import Callable, TYPE_CHECKING, Optional, List, Any, Union
import itertools
from tqdm import tqdm


from .frame import Frame


class Dataset(DatasetCore):
    def __getitem__(self, frame_number: Union[slice, int]) -> Union[DatasetCore, Frame]:
        if isinstance(frame_number, slice):
            data = self.data[frame_number]
            timestamps = self.timestamps[frame_number]
            meta = self.meta
            return Dataset(data, timestamps, meta)
        elif isinstance(frame_number, int):
            df = self.data[frame_number].compute()
            timestamp = self.timestamps[frame_number]
            return Frame(data=df, orig_file=self.meta["orig_file"], timestamp=timestamp)
        else:
            raise TypeError("Wrong type {}".format(type(frame_number).__name__))

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

    def to_file(
        self,
        file_path: Path = Path(),
    ) -> None:
        DATASET_TO_FILE["FOLDER"](self, file_path=file_path)

    def apply_pipeline(
        self,
        pipeline,
        start_frame_number: Optional[int] = 0,
        end_frame_number: Optional[int] = None,
    ) -> Any:
        """Applies a function to all, or a given range, of Frames in the dataset.

        Example:

        def pipeline1(frame: pd.Dataframe, frame_number: int):
            return frame.limit("x", 0, 1)

        test_dataset.apply_pipeline(pipeline1, 0, 10)

        Args:
            pipeline (Callable[[Frame], Frame]): A function with a chain of processings on frames.
            start_frame_number (int, optional): Frame number to start. Defaults to 0.
            end_frame_number (Optional, optional): Frame number to end. Defaults to None which corresponds to the end of the dataset.

        Returns:
            Any: depends on the pipeline functions
        """

        # use dask apply
        # maybe make a decorator for a pipeline function. The prefered way is that it can
        # return a new Dataset
        if end_frame_number is None:
            end_frame_number = len(self)
        self.data.map_partitions(
            pipeline,
        )
