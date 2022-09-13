"""" Reading pointcloud 2.

`ROS <https://www.ros.org/>`_ bagfiles.

Parts of the code are from Willow Garage, Inc.

"""

# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
# * Neither the name of Willow Garage, Inc. nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

import datetime
import itertools
import math
import struct
import sys
from pathlib import Path
from typing import Callable, Literal, Union

import numpy as np
import pandas as pd
import rosbags
import sensor_msgs.point_cloud2 as pc2  # NEED TO GET RID OF
from dask import delayed
from rich.progress import track
from sensor_msgs.msg import PointCloud2, PointField  # NEED TO GET RID OF

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)

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


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = ">" if is_bigendian else "<"

    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(
                f"Skipping unknown PointField datatype {field.datatype}",
                file=sys.stderr,
            )
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.

    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    # assert isinstance(cloud, roslib.message.Message) and cloud._type == 'sensor_msgs/PointCloud2', 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


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
    timestamps: list[datetime.datetime] = []
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
    for _ in track(range(start_frame_number, end_frame_number)):
        message = next(sliced_messages)
        timestamp = datetime.datetime.utcfromtimestamp(message.timestamp.to_sec())
        timestamps.append(timestamp)
        df = delayed(_dataframe_from_message(message, keep_zeros))
        result_list.append(df)
    return {"data": result_list, "timestamps": timestamps}
