#!/usr/bin/env python
import pandas as pd
import numpy as np
import itertools
from typing import Optional, List
import rosbag
from tqdm import tqdm
import sensor_msgs.point_cloud2 as pc2

PANDAS_TYPEMAPPING = {
    1: np.dtype("int8"),
    2: np.dtype("uint8"),
    3: np.dtype("int16"),
    4: np.dtype("uint16"),
    5: np.dtype("int32"),
    6: np.dtype("uint32"),
    7: np.dtype("float32"),
    8: np.dtype("float64"),
}


input_bagfile = "/workspaces/lidar/tests/testdata/test.bag"
bag = rosbag.Bag(input_bagfile)


def read_bag(
    bag: rosbag.Bag,
    start_frame_number: Optional[int] = 0,
    end_frame_number: Optional[int] = None,
    keep_zeros: bool = False,
    topic: str = "/os1_cloud_node/points",
) -> List:
    messages = bag.read_messages(topics=[topic])
    sliced_messages = itertools.islice(messages, start_frame_number, None)
    result_list = []
    if end_frame_number is None:
        end_frame_number = 2  # TODO fix to lenght of messages
    for frame_number in tqdm(range(start_frame_number, end_frame_number, 1)):
        message = next(sliced_messages)
        frame = dataframe_from_message(message, keep_zeros)
        result_list.append(frame)
    return result_list


def dataframe_from_message(
    message: rosbag.bag.BagMessage, keep_zeros: bool = False
) -> pd.DataFrame:
    columnnames = [item.name for item in message.message.fields]
    type_dict = {
        item.name: PANDAS_TYPEMAPPING[item.datatype] for item in message.message.fields
    }
    frame_raw = pc2.read_points(message.message)
    frame_df = pd.DataFrame(np.array(list(frame_raw)), columns=columnnames)
    frame_df = frame_df.astype(type_dict)
    if not keep_zeros:
        frame_df = frame_df[
            (frame_df["x"] != 0.0) & (frame_df["y"] != 0.0) & (frame_df["z"] != 0.0)
        ]
        frame_df["original_id"] = frame_df.index
        frame_df = frame_df.astype({"original_id": "uint32"})
        frame_df = frame_df.reset_index(drop=True)
    return frame_df


if __name__ == "__main__":
    res = read_bag(bag, 0, 2, False, "/os1_cloud_node/points")
    print(len(res))
