"""
# Dataset Class
The Dataset class which contains many frames.

For more details on how to use it please refer to the usage.ipynb Notebook for an interactive tuturial.

# Developer notes
* The important stuff happens in the __getitem__ method. Only then the rosbag is actually read with the help of
generators.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import genpy
import rosbag
from tqdm import tqdm

from .frame import Frame
from .io.dataset.bag import frame_from_message


class Dataset:
    def __init__(
        self,
        bagfile: Path,
        topic: str,
        keep_zeros: bool = False,
    ):
        """The Dataset.

        Example:
        testbag = Path().cwd().parent.joinpath("tests/testdata/test.bag")
        testset = lidar.Dataset(testbag,topic="/os1_cloud_node/points",keep_zeros=False)

        Args:
            bagfile (Path): Path to ROS bag file.
            topic (str): lidar pointcloud topic. For example "/os1_cloud_node/points"
            keep_zeros (bool, optional): Keep zero elements of points in pointclouds. Defaults to False.
        """
        self.bag = rosbag.Bag(bagfile, "r")
        """ROS bag file as a rosbag.Bag object."""
        self.orig_file = bagfile.as_posix()
        """Path to bag file."""
        if topic in self.topics_in_bag:
            self.topic = topic
        else:
            raise IOError("Topic {} not in bag.".format(topic))
        """The ROS Pointcloud2 topic of the lidar."""
        self.keep_zeros = keep_zeros
        """Option for keeping zero elements in Lidar Frames. Default is False"""
        self._len = (self.types_and_topics_in_bag.topics)[self.topic].message_count
        """Hidden length propery"""
        self._first_frame_time = self[0].timestamp.to_sec()
        """ROS Start time in the bagfile in seconds. Derifed from the first frame
        since the bag.get_start_time() can deviate form the actual time in
        the first frame.
        """
        self.start_time = self.bag.get_start_time()
        """ROS start Time in the bagfile as a float."""
        self.end_time = self.bag.get_end_time()
        """ROS End Time in the bagfile as a float."""
        self.time_step = self._calc_time_step()
        """Time step between two frames. Assumed to be constant"""

    @property
    def types_and_topics_in_bag(self):
        """Types and Topics in the original ROS bagfile as rosbag.bag.TypesAndTopicsTuple"""
        return self.bag.get_type_and_topic_info()

    @property
    def topics_in_bag(self) -> dict:
        return self.types_and_topics_in_bag.topics

    @property
    def size(self) -> int:
        """Size on disk as an int."""
        return self.bag.size

    def __len__(self) -> int:
        """Number of available frames (i.e. Lidar messages)"""
        return self._len

    def __str__(self):
        return f"Lidar Dataset with {len(self)} frame(s), from file {self.orig_file}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.orig_file})"

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, frame_number: Union[slice, int]) -> Union[List[Frame], Frame]:
        if isinstance(frame_number, slice):
            sliced_messages = self._slice_messages(frame_number)
            frame_list = []
            for message in sliced_messages:
                frame_list.append(frame_from_message(self, message))
            return frame_list
        elif isinstance(frame_number, int):
            if frame_number > 500:
                warnings.warn(
                    "Might take a long time, try using get_frame_fast instead",
                    ResourceWarning,
                )
            messages = self.bag.read_messages(topics=[self.topic])
            sliced_messages = itertools.islice(
                messages, frame_number, frame_number + 1, 1
            )
            message = next(sliced_messages)
            return frame_from_message(self, message)
        else:
            raise TypeError("Wrong type {}".format(type(frame_number).__name__))

    def has_frames(self) -> bool:
        """Check if dataset has frames.

        Returns:
            bool: ``True`` if the dataset contains frames.
        """
        return len(self) > 0

    def approximate_time_of_frame(self, frame_number: int) -> float:
        """Calcuate the approximate time of a specific frame number. Needed as input for
        get_frames_between_timestamps.

        Args:
            frame_number (int): Frame step number

        Returns:
            float: The approximate time of the frame number in seconds.
        """
        return self._first_frame_time + frame_number * self.time_step

    def get_frame_fast(self, frame_number: int) -> Frame:
        """Alternative to get a specific frame. Indented for larger bagfiles.
        This method assumes that the time between frames is constant.
        It does not replace __getitem__ fully since it relays on the method
        approximate_time_of_frame, but should produce the same result most of the times.

        Args:
            frame_number (int): frame number

        Returns:
            Frame: equivalent to Dataset[frame_number]
        """
        start_time = self.approximate_time_of_frame(frame_number)
        frame_list = self.get_frames_between_timestamps(
            start_time - 0.5 * self.time_step, start_time + 0.5 * self.time_step
        )
        return frame_list[0]

    def get_frames_between_timestamps(
        self, start_time: float, end_time: float
    ) -> List[Frame]:
        """Get all frames within specific time range

        Args:
            start_time: timestamp as float (ros posix time)
            end_time: timestamp as float (ros posix time)

        Returns:
            List of frames
        """
        messages = self.bag.read_messages(
            topics=[self.topic],
            start_time=genpy.Time.from_sec(start_time),
            end_time=genpy.Time.from_sec(end_time),
        )
        frame_list = []
        for message in messages:
            frame_list.append(frame_from_message(self, message))
        return frame_list

    def apply_pipeline(
        self,
        pipeline: Callable[[Frame, int], Frame],
        start_frame_number: Optional[int] = 0,
        end_frame_number: Optional[int] = None,
    ) -> List:
        """Applies a function to all, or a given range, of Frames in the dataset.

        Example:

        def pipeline1(frame: Frame, frame_number: int):
            return frame.limit("x", 0, 1)

        test_dataset.apply_pipeline(pipeline1, 0, 10)

        Args:
            pipeline (Callable[[Frame], Frame]): A function with a chain of processings on aframes.
            start_frame_number (int, optional): Frame number to start. Defaults to 0.
            end_frame_number (Optional, optional): Frame number to end. Defaults to None which corresponds to the end of the dataset.

        Returns:
            List: A list of results. Can be a list of Frames or other objects.
        """
        messages = self.bag.read_messages(topics=[self.topic])
        sliced_messages = itertools.islice(messages, start_frame_number, None)
        result_list = []
        if end_frame_number is None:
            end_frame_number = len(self)
        for frame_number in tqdm(range(start_frame_number, end_frame_number, 1)):
            message = next(sliced_messages)
            frame = frame_from_message(self, message)
            result_list.append(pipeline(frame, frame_number))
        return result_list

    def _slice_messages(self, frame_number: slice) -> Iterator:
        """Slice the ROS bag message generator

        Args:
            frame_number (slice): the slice position

        Raises:
            ValueError: if out of range

        Returns:
            itertools.islice: sliced message, a subset of all messages
        """
        if (frame_number.stop > frame_number.start) & (frame_number.stop <= len(self)):
            messages = self.bag.read_messages(topics=[self.topic])
            return itertools.islice(
                messages, frame_number.start, frame_number.stop, frame_number.step
            )
        else:
            raise ValueError("frame_end must be greater than frame_start and in range.")

    def _calc_time_step(self) -> float:
        """Timestep between two frames. Assumed to be constant in the whole bagfile.
        This is currently the case for mechanical spinning lidar.
        """
        return (self.end_time - self.start_time) / len(self)
