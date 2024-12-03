FROM python:3.11-slim

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
    git \
    make \
    libeigen3-dev \
    libgl1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    libpng-dev \
    libusb-1.0-0 \
    libgomp1 \
    pybind11-dev \
    pandoc \
    software-properties-common \
    mesa-utils && \
    rm -rf /var/lib/apt/lists/*

# Set up a custom prompt and colored output in the shell
RUN echo '\
    RESET="\\[\\e[0m\\]"\n\
    BOLD="\\[\\e[1m\\]"\n\
    GREEN="\\[\\e[32m\\]"\n\
    BLUE="\\[\\e[34m\\]"\n\
    export PS1="${BLUE}pointcoudset ${BLUE}${BOLD}\\w${RESET} $ "\n\
    export LS_OPTIONS="--color=auto"\n\
    eval "$(dircolors -b)"\n\
    alias ls="ls $LS_OPTIONS"\n\
    ' >> /root/.bashrc


# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
