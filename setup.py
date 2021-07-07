import codecs
import os

from setuptools import find_packages, setup


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
    name="pointcloudset",
    packages=find_packages(include=["pointcloudset", "pointcloudset*"]),
    version=get_version("pointcloudset/__init__.py"),
    author="VIRTUAL VEHICLE Research GmbH",
    author_email="thomas.goelles@v2c2.at",
    description="Analyze large datasets of point clouds recorded over time in an efficient way",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    project_urls={
        "Bug Tracker": "https://github.com/virtual-vehicle/pointcloudset/issues",
        "Source": "https://github.com/virtual-vehicle/pointcloudset",
        "Documentation": "https://virtual-vehicle.github.io/pointcloudset/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "pyntcloud",
        "open3d==0.12.0",
        "plotly",
        "fastparquet",
        "dask",
        "tqdm",
        "rospkg",
        "py3rosmsgs",
        "pycryptodomex",
        "typer",
    ],
    extras_require={"LAS": ["pylas"]},
    entry_points={
        "console_scripts": ["bag2dataset = pointcloudset.io.dataset.commandline:app"]
    },
)
