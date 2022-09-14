
FROM ghcr.io/tgoelles/python_docker:v0.3.5_py3.9


# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root 'vscode' user with sudo access. However, for Linux,
# this user's GID/UID must match your local user UID/GID to avoid permission issues
# with bind mounts. Update USER_UID / USER_GID if yours is not 1000. See
# https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID


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


# Python update conda base environment
COPY conda/environment.yml* /tmp/conda-tmp/
RUN /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml


# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

