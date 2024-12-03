from pathlib import Path

from rosbags.highlevel import AnyReader


def list_pointcloud_topics(bagfile: Path) -> list[str]:
    """Returns a list of all pointcloud topics in a ROS1 bagfile or a directory containing ROS2 mcap file."""
    with AnyReader([bagfile]) as reader:
        all_topics = reader.topics

    return [k for k, v in all_topics.items() if v.msgtype == "sensor_msgs/msg/PointCloud2"]
