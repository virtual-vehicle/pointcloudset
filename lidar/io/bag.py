"""
Routines for ROS bagfiles.
"""
import numpy as np
import pandas as pd
import rosbag
import sensor_msgs.point_cloud2 as pc2

from ..frame import Frame


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
    frame_raw = pc2.read_points(message.message)
    frame_df = pd.DataFrame(np.array(list(frame_raw)), columns=columnnames)
    if not dataset.keep_zeros:
        frame_df = frame_df[
            (frame_df["x"] != 0.0) & (frame_df["y"] != 0.0) & (frame_df["z"] != 0.0)
        ]
        frame_df = frame_df.reset_index(drop=True)
    return Frame(data=frame_df, timestamp=message.timestamp)

