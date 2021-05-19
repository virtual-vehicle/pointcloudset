from pathlib import Path

import numpy as np
import rosbag

from pointcloudset import Dataset
from pointcloudset.io.dataset.bag import _get_number_of_messages, _read_rosbag_part


def convert_bag2dir(
    bagfile: Path,
    folder_to_write: Path,
    topic: str,
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
    max_size: int = 100,
):
    bag = rosbag.Bag(bagfile.as_posix())
    max_messages = _get_number_of_messages(bag, topic)

    if end_frame_number is None:
        end_frame_number = max_messages
    if end_frame_number > max_messages:
        raise ValueError("end_frame_number to high")

    framelist = np.arange(start_frame_number, end_frame_number)

    chunks = np.array_split(framelist, int(np.ceil(len(framelist) / max_size)))
    data = []
    timestamps = []
    meta = {"orig_file": bagfile.as_posix(), "topic": topic}
    for chunk_number, chunk in enumerate(chunks):
        res = _read_rosbag_part(
            bag=bag,
            topic=topic,
            start_frame_number=chunk[0],
            end_frame_number=chunk[-1] + 1,
            keep_zeros=keep_zeros,
        )
        data = res["data"]
        timestamps = res["timestamps"]
        Dataset(data, timestamps, meta).to_file(
            folder_to_write.joinpath(f"{chunk_number}"), use_orig_filename=False
        )
