"""
Routines for ROS bagfiles.
"""
import datetime
import itertools
from pathlib import Path
from typing import Optional

import dask.dataframe as dd
import numpy as np
import pandas as pd
import rosbag
import sensor_msgs.point_cloud2 as pc2
from dask import delayed
from tqdm import tqdm

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


def dataset_from_rosbag(
    bagfile: Path,
    topic: str = "/os1_cloud_node/points",
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
) -> dd.DataFrame:
    bag = rosbag.Bag(bagfile.as_posix())
    messages = bag.read_messages(topics=[topic])
    sliced_messages = itertools.islice(messages, start_frame_number, None)
    result_list = []
    if end_frame_number is None:
        end_frame_number = 2  # TODO fix to lenght of messages
    timestamps = []
    meta = {"orig_file": bagfile.as_posix(), "topic": topic}
    for frame_number in tqdm(range(start_frame_number, end_frame_number, 1)):
        message = next(sliced_messages)
        timestamp = datetime.datetime.utcfromtimestamp(message.timestamp.to_sec())
        timestamps.append(timestamp)
        df = delayed(dataframe_from_message(message, keep_zeros))
        result_list.append(df)
    return {
        "data": dd.from_delayed(result_list),
        "timestamps": timestamps,
        "meta": meta,
    }
