import codecs
import os
import sys

from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="lidar",
    packages=[
        "lidar",
        "lidar.diff",
        "lidar.filter",
        "lidar.geometry",
        "lidar.io",
        "lidar.plot",
    ],
    version=get_version("lidar/__init__.py"),
    author="VIRTUAL VEHICLE Research GmbH",
    author_email="thomas.goelles@v2c2.at",
    description="Analyse automotive lidar data stored in ROS bagfiles",
    python_requires=">=3.7",
)
