#!/bin/bash

# Navigate to the docker subdirectory
# cd docker

# Get path of the script
script_dir=$(dirname "$(dirname "$(realpath "$0")")")
echo "Script directory: $script_dir"


# Run the container, mapping port 8888 for Jupyter and mounting volumes for data if necessary
# docker run --gpus all --runtime=nvidia --name=mvp  -v ~/master/thesis/PoinTr:/PoinTr -p 5000:80 -it --rm  --network=multi-host-network pointr-ga:configured /bin/bash

# TAG="pointr-ga:latest"
TAG="pointr-ga:configured"

docker run --gpus all \
    --name=thesis \
    -v $script_dir:/PoinTr \
    -p 5000:80 -it --rm \
    --network=multi-host-network \
    $TAG /bin/bash

    # --runtime=nvidia \
