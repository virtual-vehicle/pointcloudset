from setuptools import setup

setup(
    name="lidar",
    packages=["lidar", "lidar.file", "lidar.convert", "lidar.plot"],
    version="0.1.0",
    author="Thomas Gölles, Sarah Haas",
    author_email="thomas.goelles@v2c2.at",
    description="Analyse automotive lidar data stored in ROS bagfiles",
    python_requires=">=3.7",
)
