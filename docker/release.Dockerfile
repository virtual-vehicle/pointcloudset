FROM python:3.11-slim

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive


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


# Copy local code to the container image.
ENV PACKAGE_HOME /pointcloudset

WORKDIR $PACKAGE_HOME
ADD pointcloudset ./pointcloudset
COPY pyroject.toml ./
COPY README.rst ./
ADD doc/sphinx/source/tutorial_notebooks ./tutorial_notebooks

# install
RUN sudo /usr/local/bin/pip install  $PACKAGE_HOME

RUN /bin/bash -c "source activate base && \
    pip install  $PACKAGE_HOME"

# Make sure the environment is activated:
RUN /bin/bash -c echo "Make sure pointcloudset is installed:"
RUN /bin/bash -c python -c "from pointcloudset import Dataset"

# Export environment variables
ENV LANG=C.UTF-8

# expose dask dashboard port
EXPOSE 8787

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=

ENTRYPOINT ["/bin/bash"]
