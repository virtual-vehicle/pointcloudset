"""
The Dataset class which contains many frames.
"""

import itertools
from pathlib import Path
from typing import List, Union, Iterator

import rosbag

from .frame import Frame
from .io.bag import frame_from_message, supported_lidars


class Dataset:
    def __init__(self, bagfile: Path, lidar_name: str):
        self.bag = rosbag.Bag(bagfile, "r")
        """ROS bag file asa rosbag.Bag object"""
        self.lidar_name = lidar_name
        """Name of lidar unit"""
        self.orig_file = bagfile.as_posix()
        """Name of lidar unit"""
        self.topic = supported_lidars[lidar_name]["topic"]
        """The Pointcloud2 object of the lidar."""

    @property
    def types_and_topics_in_bag(self):
        """Types and Topics in the original ROS bagfile

        Returns:
            rosbag.bag.TypesAndTopicsTuple: All types and topics in the ROS bagfile
        """
        return self.bag.get_type_and_topic_info()

    @property
    def topics_in_bag(self) -> dict:
        """Topics in the ROS bagfile.

        Returns:
            dict: Dict with all topics in the ROS bagfile
        """
        return self.types_and_topics_in_bag.topics

    @property
    def size(self) -> int:
        """Size on disk.
        """
        return self.bag.size

    @property
    def start_time(self) -> float:
        """ROS Start time in the bagfile.
        """
        return self.bag.get_start_time()

    @property
    def end_time(self) -> float:
        """ROS End Time in the bagfile.
        """
        return self.bag.get_start_time()

    def __len__(self) -> int:
        """Number of available frames (i.e. Lidar messages)
        """
        return (self.types_and_topics_in_bag.topics)[self.topic].message_count

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
            messages = self.bag.read_messages(topics=[self.topic])
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
            raise ValueError("frame_end must be grather than frame_start and in range")
