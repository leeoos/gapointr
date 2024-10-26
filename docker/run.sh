#!/bin/bash

# Navigate to the docker subdirectory
cd docker

# Run the container, mapping port 8888 for Jupyter and mounting volumes for data if necessary
docker run --gpus all --runtime=nvidia --name=mvp  -v ~/master/thesis/PoinTr:/PoinTr -p 5000:80 -it --rm  --network=multi-host-network pointr-ga /bin/bash 
 