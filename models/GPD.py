import torch.nn as nn

# Clifford modules
from clifford_modules.mvlinear import MVLinear
from clifford_modules.mvrelu import MVReLU
from clifford_modules.gp import SteerableGeometricProductLayer
from clifford_modules.mvlayernorm import MVLayerNorm

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
    

# Clifford Algebra Neural Network
class CGEBlock(nn.Module):
    def __init__(self, algebra, in_features, out_features):
        super().__init__()

        self.layers = nn.Sequential(
            MVLinear(algebra, in_features, out_features),
            MVReLU(algebra, out_features),
            SteerableGeometricProductLayer(algebra, out_features),
            MVLayerNorm(algebra, out_features)
        )

    def forward(self, input):
        # [batch_size, in_features, 2**d] -> [batch_size, out_features, 2**d]
        return self.layers(input)


class CGEMLP(nn.Module):
    def __init__(self, algebra, in_features, hidden_features, out_features, n_layers=2):
        super().__init__()

        layers = []
        for i in range(n_layers - 1):
            layers.append(
                CGEBlock(algebra, in_features, hidden_features)
            )
            in_features = hidden_features

        layers.append(
            CGEBlock(algebra, hidden_features, out_features)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class InvariantCGENN(nn.Module):

    def __init__(self, algebra, in_features, hidden_features, out_features, restore_dim):
        super().__init__()
        self.name = "crazy test"
        self.algebra = algebra
        self.in_features = in_features
        self.cgemlp = CGEMLP(algebra, in_features, hidden_features, hidden_features)

        # self.mlp = nn.Sequential(
        #     nn.Linear(2**algebra.dim, hidden_features),
        #     nn.ReLU(),
        #     nn.Linear(hidden_features, hidden_features),
        #     nn.ReLU(),
        #     nn.Linear(hidden_features, out_features)
        # )
        self.upsampling = nn.Sequential(
            nn.Linear(hidden_features * (2**algebra.dim), in_features * algebra.dim)
        )

    def forward(self, input):
        h = self.cgemlp(input)
        # Index the hidden states at 0 to get the invariants, and let a regular MLP do the final processing.
        # print("--------")
        # print("Final step")
        # for layer in self.upsampling:
            # if isinstance(layer, nn.Linear):
                # print(f"weight shape: {layer.weight.shape}")
                # print(f"Bias shape: {layer.bias.shape}")
                # ...
        # print(f"input: {h.shape}")
        # return self.mlp(h[..., 0])
        h = h.reshape(h.size(0), -1)  # Flatten all but the first dimension
        # Pass through the linear layer
        x = self.upsampling(h)
        # Reshape to the desired output shape, keeping the batch size flexible
        x = x.reshape(-1, self.in_features, self.algebra.dim)  
        return x