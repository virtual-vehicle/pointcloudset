from setuptools import setup

setup(
    name="lidar",
    packages=["lidar", "lidar.file", "lidar.convert", "lidar.plot", "lidar.processing"],
    version="0.0.8",
    author="Thomas Gölles",
    author_email="thomas.goelles@v2c2.at",
    description="Analyse automotive lidar data",
    python_requires=">=3.7",
)
