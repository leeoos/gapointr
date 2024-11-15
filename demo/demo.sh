#! /bin/bash

python tools/inference.py cfgs/PCN_models/PoinTr.yaml ckpts/PCN_Pretrained.pth --pc demo/airplane.pcd --save_vis_img  --out_pc_root results/ 