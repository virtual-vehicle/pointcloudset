
# base docker container for pointcloudset without the installed package
# used for CI/CD and in vscode for development
FROM continuumio/miniconda3 AS base
#-------------------------------------------------------------------------------------------------------------
# Based on a template by  Microsoft Corporation.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------


# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root 'vscode' user with sudo access. However, for Linux,
# this user's GID/UID must match your local user UID/GID to avoid permission issues
# with bind mounts. Update USER_UID / USER_GID if yours is not 1000. See
# https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

COPY docker/.bashrc /root/.bashrc

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git iproute2 procps iproute2 lsb-release \
    # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # tools to access extrenal datasets
    && apt-get -y install netcat curl make wget unzip \
    && apt-get -y install gcc \
    # orca for plotly
    && apt-get install -y xvfb \
    && apt-get install -y libgtk2.0-0 \
    && apt-get install -y libasound2 \
    && apt-get install -y libxrender1 libxtst6 libxi6 libxss1 libgconf-2-4 libnss3-dev \
    && conda install -y psutil \
    && alias orca="xvfb-run orca" \
    && conda install -y xlrd  \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# install Open3D dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gdb \
    libeigen3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    libpng-dev \
    libusb-1.0-0 \
    lxde \
    mesa-utils \
    ne \
    pybind11-dev \
    software-properties-common \
    x11vnc \
    xorg-dev \
    xterm \
    xvfb && \
    rm -rf /var/lib/apt/lists/*

# Python
# Copy environment.yml (if found) to a temp locaition so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY docker/environment.yml* /tmp/conda-tmp/
RUN /opt/conda/bin/conda update -n base -c defaults conda
#RUN /opt/conda/bin/conda config --set channel_priority strict
# Update Python environment based on environment.yml (if presenet)
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp \
    # install the kernel to use with jupyter and some extensions
    && python -m ipykernel install --user --name base

# ROSPY loggging configuation file
COPY docker/python_logging.conf /etc/ros/
# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=noninteractive

########################################################################################
# Docker image including the release
FROM base AS release

# Copy local code to the container image.
ENV PACKAGE_HOME /pointcloudset

WORKDIR $PACKAGE_HOME
ADD pointcloudset ./pointcloudset
COPY setup.py ./
COPY README.rst ./
ADD doc/sphinx/source/tutorial_notebooks ./tutorial_notebooks

# install
RUN sudo /opt/conda/bin/pip install  $PACKAGE_HOME

# Make sure the environment is activated:
RUN echo "Make sure pointcloudset is installed:"
RUN python -c "from pointcloudset import Dataset"

# Export environment variables
ENV LANG=C.UTF-8

# expose dask dashboard port
EXPOSE 8787

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=
