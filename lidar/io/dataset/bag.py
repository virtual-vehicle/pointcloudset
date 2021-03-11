"""
Routines for ROS bagfiles.
"""
import numpy as np
import pandas as pd
import rosbag
import sensor_msgs.point_cloud2 as pc2

from lidar.frame import Frame

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


def frame_from_message(dataset, message: rosbag.bag.BagMessage) -> Frame:
    """Generates a frame from one ROS pointcloud2 message. Optionally with or without
    zero elements, i.e. points too close or too far away. Defined in the Dataset
    property keep_zeros.

    Args:
        dataset ([lidar.Dataset]): Dataset where the message is stored
        message (rosbag.bag.BagMessage): the message

    Returns:
        Frame: A frame with the pointcloud data
    """

    columnnames = [item.name for item in message.message.fields]
    type_dict = {
        item.name: PANDAS_TYPEMAPPING[item.datatype] for item in message.message.fields
    }
    frame_raw = pc2.read_points(message.message)
    frame_df = pd.DataFrame(np.array(list(frame_raw)), columns=columnnames)
    frame_df = frame_df.astype(type_dict)
    if not dataset.keep_zeros:
        frame_df = frame_df[
            (frame_df["x"] != 0.0) & (frame_df["y"] != 0.0) & (frame_df["z"] != 0.0)
        ]
        frame_df["original_id"] = frame_df.index
        frame_df = frame_df.astype({"original_id": "uint32"})
        frame_df = frame_df.reset_index(drop=True)
    return Frame(
        data=frame_df, orig_file=dataset.orig_file, timestamp=message.timestamp
    )
