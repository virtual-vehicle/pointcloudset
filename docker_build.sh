#!/bin/bash
docker build -f docker/Dockerfile -t tgoelles/pointcloudset_base --target base .
docker build -f docker/Dockerfile -t tgoelles/pointcloudset --target release .