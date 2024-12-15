import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ga_utils import fast_einsum, unsqueeze_like

# Clifford algebra and modules 
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra
from clifford_modules.MVLinear import MVLinear

NUM_OF_POINTS = 224 # 2048 #

class NormalizationLayer(nn.Module):
    """
    Normalization layer to scale down the elment of a multivector.
    """

    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features
        max_seq = 3000 # used to cap the parameters (Note: this is not the best approach)

        # This parameter that learn how much to scale the input data
        # in particular the how much scale the norm of input (see forward)
        self.a = nn.Parameter(torch.zeros(max_seq, algebra.n_subspaces) + init)


    def forward(self, input):
        # Small change to take in account batch size extra dimention
        assert input.shape[2] == self.in_features #
        # print(f"input.shape => {input.shape}")

        norms = torch.cat(self.algebra.norms(input), dim=-1)
        # print(f"norms.shape  before => {norms.shape}")
        s_a = torch.sigmoid(self.a)
        # print(f"s_a.shape => {s_a.shape}")
        norms = s_a[:input.shape[1], :] * (norms - 1) + 1  # interpolates between 1 and the norm
        # print(f"norms.shape  after => {norms.shape}")

        # When you see repeat_interleave usually means that
        # the same thing is repeated for each subspace
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        # print(f"norms.shape  after repeat interleave=> {norms.shape}")
        normalized = input / (norms + 1e-6)
        return normalized
    

class FullyConnectedSteerableGeometricProductLayer(nn.Module):
    def __init__(self, algebra, features, seq_lenght):
        """
        Fully connected steerable geometric product layer: a nn Module used to compute pairwise geometric products between multivectors of a same input sequence.

        Args:
            agebra: Geometric algebra object
            features: The number of features for the geometric product layer
        """
        super().__init__()
        self.algebra = algebra
        self.features = features

        # self.normalization = NormalizationLayer(algebra, features) # to change
        self.q_prj = MVLinear(algebra, seq_lenght, seq_lenght)
        self.k_prj = MVLinear(algebra, seq_lenght, seq_lenght)

    # @torch.jit.script
    def forward(self, input):
        batch, seq, dim = input.shape

        # print(f"Input shape: {input.shape}")

        # mv projection
        q = self.q_prj(input)
        k = self.k_prj(input)

        # mv normalization
        # q = self.normalization(q)
        # k = self.normalization(k)

        # Dimention adjustments
        cayley = self.algebra.cayley.to(input.device) # [dim, dim, dim]
        q_einsum = q.unsqueeze(2)  # [batch, seq, 1, dim]
        k_einsum = k.unsqueeze(1)  # [batch, 1, seq, dim]

        # Make tensor contigous in memory for performance optimization
        q_einsum = q_einsum.contiguous()
        k_einsum = k_einsum.contiguous()
        cayley = cayley.contiguous()

        # Half precision for performance optimization
        q_einsum = q_einsum.half()
        k_einsum = k_einsum.half()
        cayley = cayley.half()

        # Serve as context managers or decorators that allow regions
        # of the script to run in mixed precision
        with torch.amp.autocast('cuda'):
            output = fast_einsum(q_einsum, cayley, k_einsum)

        """
        # comment the previous 2 line and uncomment this to monitor time on gpu
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True
        ) as prof:
            with torch.amp.autocast('cuda'):
                output = fast_einsum(q_einsum, cayley, k_einsum)
                output = torch.einsum("...i,ijk,...k->...j", q_einsum, cayley, k_einsum)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        """

        # print(f"Attention output: {output.shape}")

        return output



class GeometricProductAttention(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, seq_lenght):
        """
        Self-Attention layer using geometric algebra operation.

        Args:
            algebra: Geometric algebra object
            features: The number of features for the geometric product layer
        """
        super(GeometricProductAttention, self).__init__()

        self.algebra = algebra
        self.subspaces_dims = algebra.subspaces
        self.gp_layer = FullyConnectedSteerableGeometricProductLayer(algebra, features=embed_dim, seq_lenght=seq_lenght)

        # Single projection layer to learn common propertires
        self.ffn_att_prj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Compute pairwise geometric products using the geometric product layer
        # start = time.time()
        new_mv = self.gp_layer(x)

        # apply attention score projection
        with torch.amp.autocast('cuda'):
            output = self.ffn_att_prj(new_mv)

        # end = time.time()
        # print(f"attention score computation in {end - start:.4f} seconds") # attention operation time

        return output.float()


class SelfAttentionGA(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, seq_lenght):
        super(SelfAttentionGA, self).__init__()

        self.algebra = algebra
        self.v_proj = nn.Linear(2**algebra.dim, embed_dim) #112)
        self.ga_attention = GeometricProductAttention(algebra, embed_dim, hidden_dim, seq_lenght)

    def forward(self, x):
        # x = self.algebra.embed_grade(x, 1) # shape: [B, P, 8]
        # print(f"MV embedding: {x.shape}")
        batch_size, seq_length, embed_dim = x.size() 
        v = self.v_proj(x) 
        # print(f"Value matrix shape: {v.shape}")

        # Compute attention scores using geometric product
        mv_attn = self.ga_attention(x).squeeze(-1)
        # print(f"attention scores: {mv_attn.shape}")

        attn_probs = torch.softmax(mv_attn, dim=-1)
        # print(f"attention probs: {attn_probs.shape}")

        # Apply attention to values tensor
        return torch.einsum("bqk,bvd->bqd", attn_probs, v)
    


class TransformerEncoderLayerGA(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, seq_lenght):
        super(TransformerEncoderLayerGA, self).__init__()

        self.self_attn = SelfAttentionGA(algebra, embed_dim, hidden_dim, seq_lenght)

        self.norm1 = nn.LayerNorm(embed_dim)
        # feed forward network
        self.fc_in = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # self attention and residual connection
        attn_out = self.self_attn(self.norm1(x))
        x = x + attn_out

        # feed-forward
        ff_out = self.fc_in(x)
        ff_out = self.activation(ff_out)
        ff_out = self.fc_out(ff_out)

        # residual and normalization
        x = x + ff_out
        x = self.norm2(x)

        # we are here yheeee!!!
        return x


# class PositionalEncoding(torch.nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()

#         # create a long enough position tensor
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]


class TransformerEncoderGA(nn.Module):
    def __init__(self, algebra_dim, embed_dim, hidden_dim, num_layers, seq_lenght):
        super(TransformerEncoderGA, self).__init__()

        metric = [1 for i in range(algebra_dim)]
        self.algebra = CliffordAlgebra(metric)

        self.layers = nn.ModuleList([
            TransformerEncoderLayerGA(self.algebra, embed_dim, hidden_dim, seq_lenght) 
            for _ in range(num_layers)
        ])
        # self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, x):
        # print(x.shape)
        # exit()
        x = self.algebra.embed_grade(x, 1) # geometric albebra embedding
        # x = self.pos_encoder(x)

        # Encoder layers
        for layer in self.layers:
            x = layer(x)

        return x
    

        

# # Build algebra
# algebra_dim = int(partial.shape[1])
# metric = [1 for i in range(algebra_dim)]
# print("\nBuilding the algebra...")
# algebra = CliffordAlgebra(metric)
# print(f"algebra dimention: \t {algebra.dim}")
# print(f"multivectors elements: \t {sum(algebra.subspaces)}")
# print(f"number of subspaces: \t {algebra.n_subspaces}")
# print(f"subspaces grades: \t {algebra.grades.tolist()}")
# print(f"subspaces dimentions: \t {algebra.subspaces.tolist()}")
# print("done")