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
from clifford_lib.loss.symmetryloss import SymmetryLoss
from pga_lib.pgaloss import PGALoss

# Feature extraction
from .MVFormer import TransformerEncoderGA
from .NewMVFormer import MVFormer
from models.GATR import GATrToFoldingNetAdapter
from pga_lib.pga import blade_operator

class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class PoinTrWrapper(nn.Module):
    
    def __init__(self, pointr, gafet=False):
        super().__init__()

        # PoinTr components
        self.pointr = deepcopy(pointr)

        # Geometric algebra feature extractor
        self.gafet = gafet
        if self.gafet not in ['head', 'backbone', 'fold']: 
            print(f"Invalid Geometric Algebra version: {self.gafet}\t using standard PoinTr")

        if gafet:
            # self.mvformer = TransformerEncoderGA(
            #     algebra_dim=3,  #
            #     embed_dim=8,    # fixed for 3D clifford algebra 
            #     hidden_dim=128, 
            #     num_layers=2, 
            #     seq_lenght=224,
            # )
            self.mvformer = MVFormer(
                algebra_dim=3,  # For 3D inputs
                embed_dim=8,  # Embedding size
                hidden_dim=64,  # Hidden dimension
                num_encoder_layers=4 if self.gafet == 'backbone' else 2,  # Encoder depth
                num_decoder_layers=0,  # Decoder depth
                seq_length=128 if self.gafet == 'backbone' else 224,  # Number of points
            )
            self.project_back = nn.Linear(8, 3)
            self.reduce_map = nn.Linear(384 + 1027 + 8, 384)
            self.increase_dim = nn.Sequential(
                nn.Conv1d(392, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(1024, 1024, 1)
            )
            self.foldingnet = Fold(self.pointr.trans_dim, step = self.pointr.fold_step, hidden_dim = 256)

            if self.gafet == 'head': print(f'Using head')
            if self.gafet == 'backbone':
                print(f'Using backbone')
                self.pointr.base_model.mvformer = self.mvformer
                self.pointr.base_model.project_back = self.project_back
            if self.gafet == 'fold': print(f'Using fold')


            # self.reduce_map = nn.Linear(8, 3)

            # blade = blade_operator().to('cuda')
            # self.gatr = GATrToFoldingNetAdapter(
            #     blade=blade,
            #     blade_len=blade.shape[0],
            #     hidden_dim=128,  # Hidden dim for GATr
            #     intermediate_dim=16,  # Compressed multivector dim
            #     output_dim=3,  # Desired output dim for FoldingNet
            #     n_heads=2
            # )
            # self.reduce_features = nn.Linear(
            #     in_features=4096,
            #     out_features=1024
            # )

        # Loss definition
        self.multivector_distance = MVDistance()
        self.chamfer_distance_l1 = ChamferDistanceL1()
        self.pga = PGALoss()

        # Classic loss
        self.coarse_loss = lambda x,y: self.chamfer_distance_l1(x[0], y) 
        self.fine_loss = lambda x,y: self.chamfer_distance_l1(x[1], y) 
        self.pointr_loss = lambda x,y: self.chamfer_distance_l1(x[0], y) + self.chamfer_distance_l1(x[1], y)

        # Multivector loss
        self.mvd_loss = lambda x,y: self.multivector_distance(x[0], y)
        self.mvd_reg_loss = lambda x,y: self.multivector_distance(x[0], y) + self.chamfer_distance_l1(x[0], y) + self.chamfer_distance_l1(x[1], y)

        # Symmetry loss
        self.symmetry_loss = lambda x,y: self.symmetry_loss_explicit(x, y)

        # Projective Geometric Algebra
        self.pga_loss = lambda x,y: self.pga(x, y)

        # Select loss
        self.test_loss = lambda x,y: self.chamfer_distance_l1(x[1], y)
        # self.train_loss = self.mvd_reg_loss #lambda x,y: self.chamfer_distance_l1(x[1], y) + self.multivector_distance(x[0], y)
        # self.train_loss = self.pointr_loss
        # self.train_loss = lambda x,y: self.chamfer_distance_l1(x[0], y) + self.pga(x[1], y)
        # self.train_loss = lambda x,y: 0.2*self.chamfer_distance_l1(x[1], y) + 0.8*self.pga(x[1], y)
        # self.train_loss = lambda x,y: self.chamfer_distance_l1(x[0], y) + self.chamfer_distance_l1(x[1], y) + 0.2*self.pga(x[1], y)
        self.train_loss = lambda x,y: self.chamfer_distance_l1(x[0], y) + self.chamfer_distance_l1(x[1], y) 

    
    def pga_loss_explicit(self, input, target):
        return self.pga(input, target)

    def forward(self, input):

        # Select GAPoinTr version
        if self.gafet == 'head':
            coarse, fine = self.pointr(input)
            new_fine = self.fps(fine, 224)
            new_fine = self.project_back(self.mvformer(new_fine))
            new_fine = torch.cat([fine, new_fine, input],dim=1).contiguous()
            return coarse, new_fine
        
        elif self.gafet == 'backbone':
            coarse, fine = self.pointr(input)
            return coarse, fine

        elif self.gafet == 'fold':
            # replicate pointr inference
            q, coarse_point_cloud = self.pointr.base_model(input) # B M C and B M 3
            B, M , C = q.shape

            # print(f"shape of : {q.shape}")
            
            global_feature = self.pointr.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
            global_feature = torch.max(global_feature, dim=1)[0] # B 1024
            # print(f"old global: {global_feature.shape}")

            # output_mv = self.gatr(coarse_point_cloud) #[:,:, 1:4] # extract vector part
            # pooled_mv = output_mv.mean(dim=1)  # [16, 256, 16]
            # compact_mv = pooled_mv.view(pooled_mv.size(0), -1)  # [16, 4096]
            # reduced_mv = self.reduce_features(compact_mv)  # [16, 1024]
            # ga_features = global_feature + reduced_mv  # [16, 1024]
            
            # ga_features = self.reduce_map(self.mvformer(coarse_point_cloud)) #[:,:, 1:4] # extract vector part
            ga_features = self.project_back(self.mvformer(coarse_point_cloud)) #[:,:, 1:4] # extract vector part
            # print(f"ga features shape {ga_features.shape}")
            # features_combination = torch.cat([
            #     q,
            #     ga_features
            # ], dim=-1)
            # print(f"feature combo {features_combination.shape}")
            # global_feature = self.increase_dim(features_combination.transpose(1,2)).transpose(1,2) # B M 1024
            # global_feature = torch.max(global_feature, dim=1)[0] # B 1024
            # print(f"new global: {global_feature.shape}")
            # exit()
            # print(ga_features.shape)
            # print(global_feature.unsqueeze(-2).expand(-1, M, -1).shape)
            # print(q.shape)
            # rebuild_feature = torch.cat([
            #     global_feature.unsqueeze(-2).expand(-1, M, -1),
            #     q,
            #     # coarse_point_cloud
            # ], dim=-1)  # B M 1027 + C

            rebuild_feature = torch.cat([
                global_feature.unsqueeze(-2).expand(-1, M, -1),
                q,
                # coarse_point_cloud,
                ga_features
            ], dim=-1)  # B M 1027 + C
            
            
            # rebuild_feature = torch.cat([
            #     global_feature.unsqueeze(-2).expand(-1, M, -1),
            #     q,
            #     coarse_point_cloud,
            # ], dim=-1)  # B M 1027 + C
            

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
    
    def get_model_parameters(self):
        # if self.gafet:
        #     # return list(self.pointr.foldingnet.parameters()) + list(self.pointr.reduce_map.parameters()) + list(self.gatr.parameters()) + list(self.reduce_features.parameters())
        
        #     return list(self.pointr.foldingnet.parameters()) + list(self.reduce_map.parameters()) + list(self.mvformer.parameters()) + list(self.pointr.reduce_map.parameters())
        # else:
        #     return self.parameters()
        return self.parameters()
        
        
    def fps(self, pc, num):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
        sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return sub_pc

