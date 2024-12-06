import os
import sys
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

from clifford_lib.loss.multivectorloss import MVLoss

class PoinTrWrapper(nn.Module):
    
    def __init__(self, pointr):
        super().__init__()
        self.pointr = deepcopy(pointr) # chage this to get just a piece of pointr
        self.loss_fn_0 = MVLoss([1,1,1])
        self.loss_fn_1 = ChamferDistanceL1()
        self.loss_fn = lambda x,y: 0.4*self.loss_fn_0(x[0], y) + 0.6*self.loss_fn_1(x[1], y)
        self.test_loss = lambda x,y: self.loss_fn_1(x[1], y)

    def forward(self, input):
        return self.pointr(input)
    
    def model_loss(self, output, target):
        """ In the case of fine tuning only the ch distance 
        w.r.t fine output is optimized. This could may change!"""
        return self.loss_fn(output, target)
    
    def get_model_parameters(self):
        return self.parameters()
