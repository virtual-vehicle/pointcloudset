from pathlib import Path

import numpy as np
import rosbag

from lidar import Dataset

from lidar.io.dataset.bag import get_number_of_messages, read_rosbag_part


def convert_bag2dir(
    bagfile: Path,
    folder_to_write: Path,
    topic: str = "/os1_cloud_node/points",
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
    max_size: int = 100,
) -> dict:
    bag = rosbag.Bag(bagfile.as_posix())
    max_messages = get_number_of_messages(bag, topic)

    if end_frame_number is None:
        end_frame_number = max_messages
    if end_frame_number > max_messages:
        raise ValueError("end_frame_number to high")

    framelist = np.arange(start_frame_number, end_frame_number)

    chunks = np.array_split(framelist, int(np.ceil(len(framelist) / max_size)))
    data = []
    timestamps = []
    meta = {"orig_file": bagfile.as_posix(), "topic": topic}
    chunk_number = 0
    for chunk in chunks:
        res = read_rosbag_part(
            bag,
            start_frame_number=chunk[0],
            end_frame_number=chunk[-1] + 1,
            keep_zeros=keep_zeros,
        )
        data = res["data"]
        timestamps = res["timestamps"]
        Dataset(data, timestamps, meta).to_file(
            folder_to_write.joinpath(f"{chunk_number}"), use_orig_filename=False
        )
        chunk_number = chunk_number + 1
