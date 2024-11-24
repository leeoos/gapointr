import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

# Geometric encoder
from .MVFormer import SelfAttentionGA
from .MVFormer import TransformerEncoderGA


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



class GAFeatures(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, pointr):
        super().__init__()
        # self.attention = SelfAttentionGA(algebra, embed_dim=embed_dim)
        self.transformer = TransformerEncoderGA(
            algebra, 
            embed_dim, 
            hidden_dim=hidden_dim, # fixed value for old models --> 128, 
            num_layers=2
        )
        # self.foldingnet = Fold(384, step = 8, hidden_dim = 256) 
        self.foldingnet = copy.deepcopy(pointr.foldingnet)

        self.project_back = nn.Sequential(
            nn.Linear(2**algebra.dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )


    def forward(self, pointr, pointr_parameters):

        # Extract PoinTr parameters:
        coarse_point_cloud = pointr_parameters['coarse_point_cloud']
        x = pointr_parameters['rebuild_feature']
        xyz = pointr_parameters['xyz']
        B, M, C = pointr_parameters['BMC']
        q = pointr_parameters['q']

        global_feature = pointr.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        
        ### EXPERIMENTAL PART ###
        # attention_scores = self.attention(partial)
        ga_features = self.transformer(coarse_point_cloud)
        ga_features = self.project_back(ga_features)

        # attention_scores = attention_scores.view(B, 224, 1024)
        # global_feature = torch.cat((global_feature, attention_scores), dim=1)
        #########################

        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            ga_features], dim=-1)  # B M 1027 + C

        rebuild_feature = pointr.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C

        # NOTE: foldingNet
        relative_xyz = self.foldingnet(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3

        # cat the input
        inp_sparse = self.fps(xyz, pointr.num_query)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()

        
        return rebuild_points
    

    def fps(self, pc, num):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
        sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return sub_pc
    

    # def recontruct(self, pointr, pointr_parameters, ga_features):
    #     # Extract PoinTr parameters:
    #     coarse_point_cloud = pointr_parameters['coarse_point_cloud']
    #     x = pointr_parameters['rebuild_feature']
    #     xyz = pointr_parameters['xyz']
    #     B, M, C = pointr_parameters['BMC']
    #     q = pointr_parameters['q']

    #     # PoinTr reconstruction with GA
    #     global_feature = pointr.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
    #     global_feature = torch.max(global_feature, dim=1)[0] # B 1024
    #     rebuild_feature = torch.cat([
    #         global_feature.unsqueeze(-2).expand(-1, M, -1),
    #         q,
    #         ga_features], dim=-1)  # B M 1027 + C

    #     rebuild_feature = pointr.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C

    #     # NOTE: foldingNet
    #     relative_xyz = pointr.foldingnet(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
    #     rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3

    #     # cat the input
    #     inp_sparse = self.fps(xyz, pointr.num_query)
    #     coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
    #     rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()       
    #     return rebuild_points
    


