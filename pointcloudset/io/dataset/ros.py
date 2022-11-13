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
import math
import struct
import sys
from pathlib import Path
from typing import Union, Generator, Literal

import numpy as np
import pandas as pd
from rosbags.typesys.types import sensor_msgs__msg__PointCloud2
from rosbags.rosbag1 import Reader as Reader1
from rosbags.rosbag2 import Reader as Reader2
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from dask import delayed
from rich.progress import track


_DATATYPES = {
    1: ("b", 1),
    2: ("B", 1),
    3: ("h", 2),
    4: ("H", 2),
    5: ("i", 4),
    6: ("I", 4),
    7: ("f", 4),
    8: ("d", 8),
}

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


def dataset_from_ros(
    bagfile: Path,
    topic: str,
    start_frame_number: int = 0,
    end_frame_number: int = None,
    keep_zeros: bool = False,
    ext: Literal["BAG", "ROS2"] = "BAG",
) -> Union[dict, None]:
    """Reads a Dataset from a bag file.

    Args:
        bagfile (Path): Path to bag file.
        topic (str): `ROS <https://www.ros.org/>`_ topic that should be rea
        start_frame_number (int, optional): Start pointcloud of pointcloud sequence to
            read. Defaults to 0.
        end_frame_number (int, optional): End pointcloud of pointcloud sequence to read.
            Defaults to None.
        keep_zeros (bool, optional): If ``True`` keep zeros in frames, if ``False``
            do not keep zeros in frames. Defaults to False.

    Returns:
        Union[dict, None]: Dict to generate Dataset.
    """

    data = []
    timestamps: list[datetime.datetime] = []
    meta = {"orig_file": bagfile.as_posix(), "topic": topic}

    if ext == "BAG":
        Reader = Reader1
        rosversion = 1
    elif ext == "ROS2":
        Reader = Reader2
        rosversion = 2
    else:
        raise ValueError(f"expecting BAG or ROS2 for ext got {ext}")
    with Reader(bagfile.as_posix()) as reader:
        connections = [x for x in reader.connections if x.topic == topic]

        frame = -1

        if not end_frame_number:
            end_frame_number = reader.topics[topic].msgcount

        for connection, timestamp, rawdata in track(
            reader.messages(connections=connections),
            total=end_frame_number - start_frame_number,
        ):

            frame = frame + 1
            if start_frame_number <= frame < end_frame_number:
                timestamp_datetime = datetime.datetime.fromtimestamp(timestamp * 1e-9)
                timestamps.append(timestamp_datetime)

                if rosversion == 1:
                    msg = deserialize_cdr(
                        ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                    )
                elif rosversion == 2:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                data_of_frame = delayed(
                    _dataframe_from_message(msg, keep_zeros=keep_zeros)
                )
                data.append(data_of_frame)

    return {"data": data, "timestamps": timestamps, "meta": meta}


def _dataframe_from_message(
    message: sensor_msgs__msg__PointCloud2, keep_zeros: bool = False
) -> pd.DataFrame:
    columnnames = [field.name for field in message.fields]
    type_dict = {
        item.name: PANDAS_TYPEMAPPING[item.datatype] for item in message.fields
    }
    frame_raw = _read_points(message)
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


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    """
    code from from Willow Garage, Inc.
    """
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


def _read_points(
    cloud: sensor_msgs__msg__PointCloud2, field_names=None, skip_nans=False, uvs=[]
) -> Generator:
    """
    Read points from a PointCloud2 message.
    code from from Willow Garage, Inc.

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
    assert isinstance(
        cloud, sensor_msgs__msg__PointCloud2
    ), "cloud is not a PointCloud2"
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
