from setuptools import setup

setup(
    name="lidar",
    packages=["lidar", "lidar.io", "lidar.convert", "lidar.plot", "lidar.processing"],
    version="0.0.6",
    author="Thomas Gölles",
    author_email="thomas.goelles@v2c2.at",
    description="Analyse automotive lidar data",
    python_requires=">=3.7",
)
