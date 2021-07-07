"""
Functions for `ROS <https://www.ros.org/>`_ bagfiles.
"""
import datetime
import itertools
from pathlib import Path
from typing import Callable, List, Literal, Union

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


def _dataframe_from_message(
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


def _get_number_of_messages(bag: rosbag.Bag, topic: str) -> int:
    return (bag.get_type_and_topic_info().topics)[topic].message_count


def _gen_frame_list(start_frame_number: int, end_frame_number: int, max_messages: int):
    if end_frame_number is None:
        end_frame_number = max_messages
    if end_frame_number > max_messages:
        raise ValueError("end_frame_number to high")
    return np.arange(start_frame_number, end_frame_number)


def _gen_chunks(bag, topic, start_frame_number, end_frame_number, max_size):
    max_messages = _get_number_of_messages(bag, topic)
    framelist = _gen_frame_list(start_frame_number, end_frame_number, max_messages)
    return np.array_split(framelist, int(np.ceil(len(framelist) / max_size)))


def _in_loop(
    res: dict,
    data: list,
    timestamps: list,
    folder_to_write: Path,
    meta: dict,
    chunk_number: int,
):
    data.extend(res["data"])
    timestamps.extend(res["timestamps"])


def dataset_from_rosbag(
    bagfile: Path,
    topic: str,
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
    max_size: int = 100,
    folder_to_write: Path = Path(),
    mode: Literal["internal", "cli"] = "internal",
    in_loop_function: Callable = _in_loop,
) -> Union[dict, None]:
    """Reads a Dataset from a bag file.

    Args:
        bagfile (Path): Path to bag file.
        topic (str): `ROS <https://www.ros.org/>`_ topic that should be rea
        start_frame_number (int, optional): Start pointcloud of pointcloud sequence to read. Defaults to 0.
        end_frame_number (int, optional): End pointcloud of pointcloud sequence to read.. Defaults to None.
        keep_zeros (bool, optional): If ``True`` keep zeros in frames, if ``False`` do not keep
            zeros in frames. Defaults to False.
        max_size (int, optional): Max chunk size to read from ros file at once. Defaults to 100.
        folder_to_write (Path, optional): Directly write to filder. Defaults to Path().
        mode (Literal["internal", "cli"], optional): "cli" for commandline tool. Defaults to "internal".

    Returns:
        Union[dict, None]: Dict to generate Dataset or None for cli use.
    """
    bag = rosbag.Bag(bagfile.as_posix())
    chunks = _gen_chunks(bag, topic, start_frame_number, end_frame_number, max_size)

    data = []
    timestamps: List[datetime.datetime] = []
    meta = {"orig_file": bagfile.as_posix(), "topic": topic}

    for chunk_number, chunk in enumerate(chunks):
        res = _read_rosbag_part(
            bag=bag,
            topic=topic,
            start_frame_number=chunk[0],
            end_frame_number=chunk[-1] + 1,
            keep_zeros=keep_zeros,
        )
        in_loop_function(res, data, timestamps, folder_to_write, meta, chunk_number)
    if mode == "internal":
        return {
            "data": data,
            "timestamps": timestamps,
            "meta": meta,
        }
    else:
        return None


def _read_rosbag_part(
    bag: rosbag.bag,
    topic: str,
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
) -> dict:
    messages = bag.read_messages(topics=[topic])
    sliced_messages = itertools.islice(messages, start_frame_number, None)
    result_list = []
    max_messages = _get_number_of_messages(bag, topic)
    if end_frame_number is None:
        end_frame_number = max_messages
    if end_frame_number > max_messages:
        raise ValueError("end_frame_number to high")
    timestamps = []
    for _ in tqdm(range(start_frame_number, end_frame_number)):
        message = next(sliced_messages)
        timestamp = datetime.datetime.utcfromtimestamp(message.timestamp.to_sec())
        timestamps.append(timestamp)
        df = delayed(_dataframe_from_message(message, keep_zeros))
        result_list.append(df)
    return {"data": result_list, "timestamps": timestamps}
