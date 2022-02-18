# A script to build docker images localy for testing, without the need of gh actions

docker build --rm -f "docker/base.Dockerfile" -t tgoelles/pointcloudset_base:v0.3.4 "."

#docker build --rm -f "docker/release.Dockerfile" -t tgoelles/pointcloudset:v0.3.4 "."