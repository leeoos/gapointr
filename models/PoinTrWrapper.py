import os
import sys
import torch 
import torch.nn as nn
from copy import deepcopy

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

class PoinTrWrapper(nn.Module):
    
    def __init__(self, pointr):
        super().__init__()
        self.model = deepcopy(pointr) # chage this to get just a piece of pointr
    
    def forward(self, input):
        return self.model(input)