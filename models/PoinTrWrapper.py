import os
import sys
from pointnet2_pytorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
import torch 
import torch.nn as nn
from copy import deepcopy

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# Loss functions
from extensions.chamfer_dist import (
    ChamferDistanceL1, 
    ChamferDistanceL2
)

from clifford_lib.loss.multivectordistance import MVDistance
from .MVFormer import TransformerEncoderGA

class PoinTrWrapper(nn.Module):
    
    def __init__(self, pointr, gafte=False):
        super().__init__()

        # PoinTr components
        self.pointr = deepcopy(pointr)

        # Geometric algebra feature extractor
        self.gafte = gafte
        if gafte:
            self.mvformer = TransformerEncoderGA(
                algebra_dim=3,  #
                embed_dim=8,    # fixed for 3D clifford algebra 
                hidden_dim=256, 
                num_layers=2, 
                seq_lenght=224,
            )
            self.project_back = nn.Linear(in_features=8, out_features=3)

        # Loss definition
        self.multivector_distance = MVDistance()
        self.chamfer_distance_l1 = ChamferDistanceL1()

        # Classic loss
        self.coarse_loss = lambda x,y: self.chamfer_distance_l1(x[0], y) 
        self.fine_loss = lambda x,y: self.chamfer_distance_l1(x[1], y) 
        self.pointr_loss = lambda x,y: self.chamfer_distance_l1(x[0], y) + self.chamfer_distance_l1(x[1], y)

        # Multivector loss
        self.mvd_loss = lambda x,y: self.multivector_distance(x[0], y)
        self.mvd_reg_loss = lambda x,y: self.multivector_distance(x[0], y) + self.chamfer_distance_l1(x[0], y) + self.chamfer_distance_l1(x[1], y)

        # Select loss
        self.test_loss = lambda x,y: self.chamfer_distance_l1(x[1], y)
        self.train_loss = lambda x,y: self.chamfer_distance_l1(x[0], y) + self.chamfer_distance_l1(x[1], y) + self.multivector_distance(x[0], y)
        # self.train_loss = self.pointr_loss

    def forward(self, input):
        if self.gafte:
            # replicate pointr inference
            q, coarse_point_cloud = self.pointr.base_model(input) # B M C and B M 3
            B, M ,C = q.shape

            ga_features = self.project_back(self.mvformer(coarse_point_cloud)) #[:,:, 1:4] # extract vector part
            global_feature = self.pointr.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
            global_feature = torch.max(global_feature, dim=1)[0] # B 1024

            rebuild_feature = torch.cat([
                global_feature.unsqueeze(-2).expand(-1, M, -1),
                q,
                ga_features], dim=-1)  # B M 1027 + C
            
            # print(f"Debug: rebuild_features: {ga_features.shape}")
            # exit()
            
            rebuild_feature = self.pointr.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C
            relative_xyz = self.pointr.foldingnet(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3

            # cat the input
            inp_sparse = self.fps(input, self.pointr.num_query)
            coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
            rebuild_points = torch.cat([rebuild_points, input],dim=1).contiguous()
            ret = (coarse_point_cloud, rebuild_points)
            return ret

        else:
            return self.pointr(input)
    
    # def model_loss(self, output, target):
    #     """ In the case of fine tuning only the ch distance 
    #     w.r.t fine output is optimized. This could may change!"""
    #     return self.train_loss(output, target)
    
    def get_model_parameters(self):
        if self.gafte:
            return list(self.pointr.foldingnet.parameters()) + list(self.pointr.reduce_map.parameters()) + list(self.mvformer.parameters())
        else:
            return self.pointr.parameters()
        
    def fps(self, pc, num):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
        sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return sub_pc

