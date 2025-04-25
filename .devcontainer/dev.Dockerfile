FROM ghcr.io/astral-sh/uv:bookworm-slim

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

ARG USER_NAME=vscode
ARG USER_HOME=/home/${USER_NAME}
ARG USER_ID=1000
ARG USER_GECOS=vscode

# Create user
RUN adduser --home "${USER_HOME}" --uid "${USER_ID}" --gecos "${USER_GECOS}" --disabled-password "${USER_NAME}"


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

RUN echo '\
    RESET="\\[\\e[0m\\]"\n\
    BOLD="\\[\\e[1m\\]"\n\
    GREEN="\\[\\e[32m\\]"\n\
    BLUE="\\[\\e[34m\\]"\n\
    export PS1="${BLUE}backend ${BLUE}${BOLD}\\w${RESET} $ "\n\
    export LS_OPTIONS="--color=auto"\n\
    eval "$(dircolors -b)"\n\
    alias ls="ls $LS_OPTIONS"\n\
    ' >> /home/${USER_NAME}/.bashrc

USER ${USER_NAME}

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=noninteractive

