#!/bin/bash

# Ensure BuildKit is enabled
export DOCKER_BUILDKIT=1
cd "$(dirname "$0")/.."

# Use Docker Buildx to build the image
while true; do
    case "$1" in
        -c) docker buildx build -f Dockerfile --build-arg BUILDKIT_INLINE_CACHE=1 -t pointr-ga --cache-from pointr-ga:latest . ; break;;
        -p) echo "pruning" ; docker builder prune -f; docker buildx build -f docker/Dockerfile -t pointr-ga .; break;;
        *) break ;;
    esac
done

if [ -z "$1" ]; then
    docker buildx build -f Dockerfile -t pointr-ga .
fi
