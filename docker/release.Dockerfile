FROM ghcr.io/astral-sh/uv:0.11.8-debian-slim

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive


# Copy local code to the container image.
ENV PACKAGE_HOME=/pointcloudset

WORKDIR $PACKAGE_HOME
ADD src ./src
COPY pyproject.toml ./
COPY README.rst ./
ADD doc/sphinx/source/tutorial_notebooks ./tutorial_notebooks

# install
RUN uv sync --no-dev

# Make sure the everything is installed ok
RUN /bin/bash -c echo "Make sure pointcloudset is installed:"
RUN /bin/bash -c "uv run python -c 'import pointcloudset; print(pointcloudset.__version__); from pointcloudset import Dataset'"
RUN /bin/bash -c "uv run pointcloudset --help || (echo 'CLI not installed' && exit 1)"

# Export environment variables
ENV LANG=C.UTF-8

# expose dask dashboard port
EXPOSE 8787

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=

ENTRYPOINT ["/bin/bash"]
