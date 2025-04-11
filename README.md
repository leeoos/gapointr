

# Point Cloud Completion with Geometry-Aware Transformers

This repository contains the implementation of the paper "PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers" by Xumin Yu, Yongming Rao, Ziyi Wang, Zuyan Liu, Jiwen Lu, and Jie Zhou.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

## Installation
The first step is to build the docker image. You can do this by running the following command in the root directory of the repository:

```sh
bash gapointr/docker/build.sh
```
Then run the container with the following command:

```sh
bash gapointr/docker/run.sh
```
To install the necessary dependencies inside the container, run the following script:

```sh
source install.sh
```

After this export the container:
```sh
docker commit $(docker ps -q --filter ancestor=pointr-ga) pointr-ga:configured
```
Then decomment line 15 of docker/run.sh
