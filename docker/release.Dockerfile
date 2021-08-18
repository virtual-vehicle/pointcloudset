########################################################################################
# Docker image including the release
FROM tgoelles/pointcloudset_base:v0.3.1

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

ENTRYPOINT [ "/bin/bash" ]