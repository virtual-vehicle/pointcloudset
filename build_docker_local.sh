# A script to build docker images localy for testing, without the need of gh actions

docker build --rm -f "docker/release.Dockerfile" -t tgoelles/pointcloudset:v0.10.1 "."

#docker build --rm -f "docker/release.Dockerfile" -t tgoelles/pointcloudset:v0.3.4 "."