import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
from utils.ga_utils import fast_einsum, unsqueeze_like

# Clifford algebra and modules 
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra
from clifford_modules.MVLinear import MVLinear


class FullyConnectedSteerableGeometricProductLayer(nn.Module):
    def __init__(self, algebra, features, seq_length):
        """
        Fully connected steerable geometric product layer for pairwise geometric products.

        Args:
            algebra: Clifford Algebra object.
            features: Number of features for geometric product layer.
            seq_length: Sequence length of input.
        """
        super().__init__()
        self.algebra = algebra
        self.features = features

        self.normalization = nn.LayerNorm(features)  # Using LayerNorm for improved stability
        self.q_prj = MVLinear(algebra, seq_length, seq_length)
        self.k_prj = MVLinear(algebra, seq_length, seq_length)

    def forward(self, input):
        batch, seq, dim = input.shape

        q = self.normalization(self.q_prj(input))
        k = self.normalization(self.k_prj(input))

        cayley = self.algebra.cayley.to(input.device).contiguous()
        q_einsum = q.unsqueeze(2).contiguous().half()
        k_einsum = k.unsqueeze(1).contiguous().half()
        cayley = cayley.half()

        with torch.amp.autocast('cuda'):
            output = fast_einsum(q_einsum, cayley, k_einsum)

        return output.float()

class GeometricProductAttention(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, seq_length):
        super().__init__()
        self.algebra = algebra
        self.gp_layer = FullyConnectedSteerableGeometricProductLayer(algebra, features=embed_dim, seq_length=seq_length)
        self.ffn_att_prj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        new_mv = self.gp_layer(x)
        with torch.amp.autocast('cuda'):
            output = self.ffn_att_prj(new_mv)
        return self.dropout(output.float())

class SelfAttentionGA(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, seq_length):
        super().__init__()
        self.algebra = algebra
        self.v_proj = nn.Linear(2**algebra.dim, embed_dim)
        self.ga_attention = GeometricProductAttention(algebra, embed_dim, hidden_dim, seq_length)

    def forward(self, x, memory=None):
        batch_size, seq_length, embed_dim = x.size()
        v = self.v_proj(x)

        if memory is not None:
            x = x + memory  # Combine input with memory if provided

        mv_attn = self.ga_attention(x).squeeze(-1)
        attn_probs = torch.softmax(mv_attn, dim=-1)
        return torch.einsum("bqk,bvd->bqd", attn_probs, v)

class TransformerEncoderLayerGA(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, seq_length):
        super().__init__()
        self.self_attn = SelfAttentionGA(algebra, embed_dim, hidden_dim, seq_length)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc_in = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn_out = self.self_attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.fc_out(self.activation(self.fc_in(x)))
        x = x + self.dropout(ff_out)
        return self.norm2(x)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderGA(nn.Module):
    def __init__(self, algebra_dim, embed_dim, hidden_dim, num_layers, seq_length):
        super().__init__()
        metric = [1 for _ in range(algebra_dim)]
        self.algebra = CliffordAlgebra(metric)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerGA(self.algebra, embed_dim, hidden_dim, seq_length)
            for _ in range(num_layers)
        ])
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, x):
        x = self.algebra.embed_grade(x, 1)  # Geometric Algebra embedding
        x = self.pos_encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class TransformerDecoderLayerGA(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, seq_length):
        super().__init__()
        self.self_attn = SelfAttentionGA(algebra, embed_dim, hidden_dim, seq_length)
        self.cross_attn = SelfAttentionGA(algebra, embed_dim, hidden_dim, seq_length)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.fc_in = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(self.norm1(tgt))
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.cross_attn(self.norm2(tgt), memory)
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.fc_out(self.activation(self.fc_in(tgt)))
        tgt = tgt + self.dropout(tgt2)
        return self.norm3(tgt)

class TransformerDecoderGA(nn.Module):
    def __init__(self, algebra_dim, embed_dim, hidden_dim, num_layers, seq_length):
        super().__init__()
        metric = [1 for _ in range(algebra_dim)]
        self.algebra = CliffordAlgebra(metric)
        self.layers = nn.ModuleList([
            TransformerDecoderLayerGA(self.algebra, embed_dim, hidden_dim, seq_length)
            for _ in range(num_layers)
        ])
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, tgt, memory):
        tgt = self.algebra.embed_grade(tgt, 1)  # Geometric Algebra embedding
        tgt = self.pos_encoder(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt

class MVFormer(nn.Module):
    def __init__(self, algebra_dim, embed_dim, hidden_dim, num_encoder_layers, num_decoder_layers, seq_length):
        super().__init__()
        self.decoder_layers = num_decoder_layers
        self.encoder = TransformerEncoderGA(algebra_dim, embed_dim, hidden_dim, num_encoder_layers, seq_length)

        if self.decoder_layers > 0:
            self.decoder = TransformerDecoderGA(algebra_dim, embed_dim, hidden_dim, num_decoder_layers, seq_length)

    def forward(self, src, tgt=None):

        if self.decoder_layers > 0:
            if not tgt: 
                raise Exception(f"Invalid target: {tgt}")
            memory = self.encoder(src)
            output = self.decoder(tgt, memory)
        else:
            output = self.encoder(src)
        return output


def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS) for point cloud data.
    Args:
        points: Tensor of shape [batch_size, num_points, 3]
        num_samples: Number of points to sample
    Returns:
        Tensor of sampled points of shape [batch_size, num_samples, 3]
    """
    batch_size, num_points, _ = points.shape
    centroids = torch.zeros(batch_size, num_samples, dtype=torch.long).to(points.device)
    distances = torch.ones(batch_size, num_points).to(points.device) * 1e10
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long).to(points.device)
    
    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[torch.arange(batch_size), farthest].unsqueeze(1)  # [batch_size, 1, 3]
        dist = torch.sum((points - centroid) ** 2, dim=-1)  # Euclidean distance
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, dim=-1)[1]
    return points[torch.arange(batch_size).unsqueeze(1), centroids]

# Incorporating FPS into forward pass:
class GAPointCloudReconstructionTransformer(MVFormer):
    def __init__(self, algebra_dim, embed_dim, hidden_dim, num_encoder_layers, num_decoder_layers, seq_length, num_output_points):
        super().__init__(algebra_dim, embed_dim, hidden_dim, num_encoder_layers, num_decoder_layers, seq_length)
        self.num_output_points = num_output_points
        self.reconstruction_layer = nn.Linear(embed_dim, 3)  # Map latent space back to 3D coordinates

    def forward(self, src):
        memory = self.encoder(src)  # Encode partial point cloud
        print(memory.shape)
        queries = farthest_point_sampling(src, self.num_output_points)  # Sample queries from partial input
        output = self.decoder(queries, memory)  # Decode to reconstruct missing points
        reconstructed_points = self.reconstruction_layer(output)  # Map to 3D space
        return reconstructed_points


if __name__ == "__main__":

    model = GAPointCloudReconstructionTransformer(
        algebra_dim=3,  # For 3D inputs
        embed_dim=8,  # Embedding size
        hidden_dim=128,  # Hidden dimension
        num_encoder_layers=4,  # Encoder depth
        num_decoder_layers=4,  # Decoder depth
        seq_length=144,  # Number of points
        num_output_points=144
    ).to('cuda')

    input_points = torch.rand(16, 144, 3).to('cuda')

    output = model(input_points)
    print(output.shape)