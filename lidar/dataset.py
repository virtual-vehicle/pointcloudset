"""
The Dataset class which contains many frames. 

For more details on how to use it please refer to the usage.ipynb Notebook for an interactive tuturial.
"""

import itertools
from pathlib import Path
from typing import Iterator, List, Union

import genpy
import rosbag

from .file.bag import frame_from_message
from .frame import Frame


class Dataset:
    def __init__(
        self,
        bagfile: Path,
        topic: str,
        timerange: tuple = (None),
        keep_zeros: bool = False,
    ):
        """The Dataset.

        Example:
        testbag = Path().cwd().parent.joinpath("tests/testdata/test.bag")
        testset = lidar.Dataset(testbag,topic="/os1_cloud_node/points",keep_zeros=False)

        Args:
            bagfile (Path): Path to ROS bag file.
            topic (str): lidar pointcloud topic. For example "/os1_cloud_node/points"
            timerange: Only messages between timerange[0] and timerange[1] will be read from file.
            keep_zeros (bool, optional): Keep zero elements. Defaults to False.
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
        self.timerange = timerange
        """Messages between start and end time will be read from the bag file."""
        self.keep_zeros = keep_zeros
        """Option for keeping zero elements in Lidar Frames. Default is False"""

    @property
    def types_and_topics_in_bag(self):
        """Types and Topics in the original ROS bagfile as rosbag.bag.TypesAndTopicsTuple
        """
        return self.bag.get_type_and_topic_info()

    @property
    def topics_in_bag(self) -> dict:
        """Topics in the ROS bagfile as a dict.
        """
        return self.types_and_topics_in_bag.topics

    @property
    def size(self) -> int:
        """Size on disk as an int.
        """
        return self.bag.size

    @property
    def start_time(self) -> float:
        """ROS Start time in the bagfile as a float.
        """
        return self.bag.get_start_time()

    @property
    def end_time(self) -> float:
        """ROS End Time in the bagfile as a float.
        """
        return self.bag.get_end_time()

    def __len__(self) -> int:
        """Number of available frames (i.e. Lidar messages)
        """
        if self.timerange is None and self.keep_zeros is True:
            return (self.types_and_topics_in_bag.topics)[self.topic].message_count
        else:
            l = sum(1 for m in self)
            return l

    def __str__(self):
        return f"Lidar Dataset with {len(self)} frame(s), from file {self.orig_file}"

    def __getitem__(self, frame_number: Union[int, slice]) -> Union[Frame, List[Frame]]:
        if isinstance(frame_number, slice):
            sliced_messages = self._slice_messages(frame_number)
            frame_list = []
            for message in sliced_messages:
                frame_list.append(frame_from_message(self, message))
            return frame_list
        elif isinstance(frame_number, int):
            if self.timerange is None:
                messages = self.bag.read_messages(topics=[self.topic])
            else:
                messages = self.bag.read_messages(
                    topics=[self.topic],
                    start_time=genpy.Time.from_sec(self.timerange[0]),
                    end_time=genpy.Time.from_sec(self.timerange[1]),
                )

            sliced_messages = itertools.islice(
                messages, frame_number, frame_number + 1, 1
            )
            message = next(sliced_messages)
            return frame_from_message(self, message)
        else:
            raise TypeError("Wrong type {}".format(type(frame_number).__name__))

    def has_frames(self):
        """Check if dataset has frames.

        Returns:
            bool: ``True`` if the dataset contains frames.
        """
        return len(self) > 0

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
