import torch.nn as nn

# Clifford modules
from clifford_modules.mvlinear import MVLinear
from clifford_modules.mvrelu import MVReLU

class GPD(nn.Module):
    def __init__(self, algebra, input_dim=3, hidden_dim=64, output_dim=3):
        super(GPD, self).__init__()
        self.name = 'Geometric_algebra_Point_cloud_Deformer'
        
        # self.cgemlp = CGEBlock(algebra, input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            MVLinear(algebra, input_dim, hidden_dim),
            MVReLU(algebra, hidden_dim),
            MVLinear(algebra, hidden_dim, hidden_dim),
            MVReLU(algebra, hidden_dim),
            MVLinear(algebra, hidden_dim, output_dim)
        )
        # Projecting multivectors to points
        self.prj = nn.Linear(in_features=2**algebra.dim, out_features=1)  

    def forward(self, input):
        h = self.mlp(input)
        # Index the hidden states at 0 to get the invariants, and let a regular MLP do the final processing.
        # print(h.shape)
        return self.prj(h).squeeze()
    


# Simple folding model (standard MLP)