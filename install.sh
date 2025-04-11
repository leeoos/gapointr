#! /bin/bash

echo "run this with source install.sh from the root directory"

ROOT=$(pwd)

echo $ROOT

# Chamfer Distance
cd $ROOT/extensions/chamfer_dist
python setup.py install

# EMD
cd $ROOT/extensions/emd
python setup.py install

# Cubic Feature Sampling
cd $ROOT/extensions/cubic_feature_sampling
python setup.py install

# Gridding & Gridding Reverse
cd $ROOT/extensions/gridding
python setup.py install

# Gridding Loss
cd $ROOT/extensions/gridding_loss
python setup.py install

cd $ROOT/pointnet2_pytorch
pip install ./pointnet2_ops_lib

cd $ROOT
echo $ROOT

# Packeage correction (it may be not necessary when docker is rebuild)
# pip uninstall pyg-lib torch-sparse
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric --no-cache

echo "Done!"
