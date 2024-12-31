import os 
import sys
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange

# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../'))
from pga_lib.pga import *

# # PGA
# from pga import *

blade = blade_operator().to('cuda')
blade_len = blade.shape[0]

class EquiLinearLayer(nn.Module):
    """
    Equivariant Linear Layer.

    Args:
        input_mv_channels (int): number of input channels in the multivector
        hidden_mv_dim (int): number of output channels in the multivector
        blade (torch.Tensor): blade tensor representing the geometric entity
        blade_len (int): length of the blade tensor

    Attributes:
        blade (torch.Tensor): blade tensor representing blade operator
        weights (nn.Parameter): learnable weights for the linear layer

    Methods:
        forward(x): computes the forward pass of the equivariant linear layer
    """

    def __init__(self,input_mv_channels, hidden_mv_dim, blade, blade_len):
        super(EquiLinearLayer,self).__init__()
        self.blade = blade
        self.weights = nn.Parameter(
            torch.rand(hidden_mv_dim, input_mv_channels, blade_len, device='cuda')
         )

    def forward(self,x):
        # print(f"Input shape: {x.shape}")
        # print(f"Weights shape: {self.weights.shape}")
        # print(f"Blade shape: {self.blade.shape}")
        # exit()
        output_mv = torch.einsum(
            "j i b, b x y, ... i x -> ... j y",
            self.weights,
            self.blade,
            x
         )
        return output_mv
    

class EquilinearNormLayer(nn.Module):
    """
    Custom layer for normalizing multivectors in an equivariant neural network.

    Args:
        faster (bool): flag indicating whether to use the faster inner product calculation

    Attributes:
        faster (bool): flag indicating whether to use the faster inner product calculation

    Methods:
        forward(x): computes the forward pass to normalize multivectors in an equivariant manner
    """

    def __init__(self, faster = True):
        super(EquilinearNormLayer,self).__init__()
        self.faster = faster

    def forward(self, x):
        mv_inner_product = faster_inner_product(x,x)
        squared_norms = torch.mean(mv_inner_product, dim=-2, keepdim=True)

        # Rescale inputs
        outputs = x / torch.sqrt(squared_norms)

        return outputs


def prepare_qkv(q,k,v,n_heads,hidden_dim):
    """
    Prepares the query (q), key (k), and value (v) tensors for the
    attention mechanism in a transformer, with the inner product.

    Args:
        q (torch.Tensor): query tensor
        k (torch.Tensor): key tensor
        v (torch.Tensor): value tensor
        n_heads (int): number of attention heads
        hidden_dim (int): hidden dimension

    Returns:
        (torch.Tensor): modified query tensor
        (torch.Tensor): modified key tensor
        (torch.Tensor): modified value tensor
    """

    q = rearrange(
        q,
        "... items (hidden_dim n_heads) mv -> ... n_heads items hidden_dim mv",
        n_heads = n_heads,
        hidden_dim = hidden_dim
     )

    k = rearrange(
        k,
        "... n_items (hidden_dim 1) mv_dim -> ... 1 n_items hidden_dim mv_dim"
     )

    v = rearrange(
        v,
        "... n_items (hidden_dim 1) mv_dim -> ... 1 n_items hidden_dim mv_dim"
     )

    guidance_matrix = get_guidance_matrix().to('cuda')
    reverse_op = reverse_operator()
    inner_product_mask = (torch.diag(guidance_matrix[0]) * reverse_op).bool()

    ranges = get_coordinates_range()
    index_product_idxs = list(range(ranges[0][0],ranges[-1][-1]+1))
    index_product_coordinates = [
        coord for coord,
        keep in zip(index_product_idxs, inner_product_mask) if keep
     ]

    q = rearrange(
        q[..., index_product_coordinates],
        "... c x -> ... (c x)"
     )
    k = rearrange(
        k[..., index_product_coordinates],
        "... c x -> ... (c x)"
     )

    v = rearrange(
        v,
        "... c x -> ... (c x)"
     )

    return q,k,v

class GeometricAttentionLayer(nn.Module):
    """
    Geometric equivariant attention

    Args:
        mv_channels (int): number of channels in the input multivector
        hidden_dim (int): hidden dimension for attention calculations
        out_channels (int): number of channels in the output multivector
        blade (torch.Tensor): blade tensor representing the geometric entity
        blade_len (int): length of the blade tensor
        n_heads (int): number of attention heads

    Attributes:
        n_heads (int): number of attention heads
        mv_channels (int): number of channels in the input multivector
        hidden_dim (int): hidden dimension for attention calculations
        out_channels (int): number of channels in the output multivector
        q (EquiLinearLayer): layer for queries
        k (EquiLinearLayer): layer for keys
        v (EquiLinearLayer): layer for values
        output_projection (EquiLinearLayer): layer for output projection

    Methods:
        forward(x): computes the forward pass of the geometric attention layer
    """
    def __init__(
        self,mv_channels,hidden_dim,out_channels,blade,blade_len,n_heads
    ):
        super(GeometricAttentionLayer,self).__init__()
        self.n_heads = n_heads
        self.mv_channels = mv_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.q = EquiLinearLayer(
            input_mv_channels = mv_channels,
            hidden_mv_dim = hidden_dim * n_heads,
            blade = blade,
            blade_len = blade_len
         )

        self.k = EquiLinearLayer(
            input_mv_channels = mv_channels,
            hidden_mv_dim = hidden_dim,
            blade = blade,
            blade_len = blade_len
         )

        self.v = EquiLinearLayer(
            input_mv_channels = mv_channels,
            hidden_mv_dim = hidden_dim,
            blade = blade,
            blade_len = blade_len
         )

        self.output_projection = EquiLinearLayer(
            input_mv_channels = hidden_dim * n_heads,
            hidden_mv_dim = out_channels,
            blade = blade,
            blade_len = blade_len
         )

    def forward(self,x):
        q,k,v = prepare_qkv(
            self.q(x),
            self.k(x),
            self.v(x),
            self.n_heads,
            self.hidden_dim
         )


        attention = scaled_dot_product_attention(q, k, v)
        attention_mv = rearrange(
            attention[..., : self.hidden_dim * x.shape[-1]],
            "... (c x) -> ...  c x", x=x.shape[-1]
         )
        attention_mv = rearrange(
            attention_mv,
            "... heads items hidden_dim x -> ... items (heads hidden_dim) x"
         )

        outputs_mv = self.output_projection(attention_mv)

        return outputs_mv
    
class GeometricBilinearLayer(nn.Module):
    """
    Geometric bilinear layer.

    Args:
        in_mv_channels (int): number of channels in the input multivector
        hidden_mv_channels (int): number of channels in the hidden multivector

    Attributes:
        geom_linear_1 (EquiLinearLayer): layer for geometric product operand 1
        geom_linear_2 (EquiLinearLayer): layer for geometric product operand 2
        join_linear_1 (EquiLinearLayer): layer for join operand 1
        join_linear_2 (EquiLinearLayer): layer for join operand 2

    Methods:
        forward(x, ref): computes the forward pass of the geometric bilinear layer
    """

    def __init__(self,in_mv_channels,hidden_mv_channels):
        super(GeometricBilinearLayer,self).__init__()

        self.geom_linear_1 = EquiLinearLayer(
            input_mv_channels = in_mv_channels,
            hidden_mv_dim = hidden_mv_channels // 2,
            blade = blade,
            blade_len = blade_len
         )

        self.geom_linear_2 = EquiLinearLayer(
            input_mv_channels = in_mv_channels,
            hidden_mv_dim = hidden_mv_channels // 2,
            blade = blade,
            blade_len = blade_len
         )

        self.join_linear_1 = EquiLinearLayer(
            input_mv_channels = in_mv_channels,
            hidden_mv_dim = hidden_mv_channels // 2,
            blade = blade,
            blade_len = blade_len
         )

        self.join_linear_2 = EquiLinearLayer(
            input_mv_channels = in_mv_channels,
            hidden_mv_dim = hidden_mv_channels // 2,
            blade = blade,
            blade_len = blade_len
         )

    def forward(self,x,ref):
        geom_linear_1 = self.geom_linear_1(x)
        geom_linear_2 = self.geom_linear_2(x)
        geom_product = geometric_product(
            geom_linear_1, geom_linear_2
        ) * 10e-6

        join_linear_1 = self.join_linear_1(x)
        join_linear_2 = self.join_linear_2(x)
        equi_join = join(join_linear_1, join_linear_2, ref)
        outputs_mv = torch.cat((geom_product, equi_join), dim=-2)

        return outputs_mv
    

class GatedRELU(nn.Module):
    """
    Gated GeLU activation

    Attributes:
        gelu (nn.GELU): GELU activation function

    Methods:
        forward(x): computes the forward pass of the Gated GeLU layer
    """

    def __init__(self):
        super(GatedRELU,self).__init__()
        self.relu = nn.ReLU()

    def forward(self,x):
        gates = x[...,[0]]
        weights = self.relu(gates)
        outputs = weights * x
        return outputs
    

class GATrNet(nn.Module):
    """
    GATr neural network.

    Args:
        in_channels (int): Number of input channels in the data
        blade (torch.Tensor): Blade tensor for EquilinearLayer operations
        blade_len (int): Length of the blade tensor
        hidden_dim (int): Hidden dimension of the EquilinearLayer

    Attributes:
        enter_equilinear (EquiLinearLayer): EquilinearLayer for initial projection
        attention_equinorm (EquilinearNormLayer): EquilinearNormLayer for attention block normalization
        geometric_attention (GeometricAttentionLayer): GeometricAttentionLayer for attention block
        bilinear_equinorm (EquilinearNormLayer): EquilinearNormLayer for bilinear block normalization
        geometric_bilinear (GeometricBilinearLayer): GeometricBilinearLayer for bilinear block
        gated_gelu (GatedGELU): Gated GELU activation function
        gelu_equilinear (EquiLinearLayer): EquilinearLayer for final GELU operation
        final_equilinear (EquiLinearLayer): Final EquilinearLayer for output projection
        vectorizer (nn.Linear): Linear layer for vectorization

    Methods:
        forward(x): computes the forward pass of the GATrNet
    """

    def __init__(self, in_channels, blade, blade_len, hidden_dim, n_heads):
        super(GATrNet, self).__init__()

        # Entering Equilinear Layer
        self.enter_equilinear = EquiLinearLayer(
            input_mv_channels = in_channels,
            hidden_mv_dim = hidden_dim,
            blade = blade,
            blade_len = blade_len
         )

        # Attention block
        self.attention_equinorm = EquilinearNormLayer()

        self.geometric_attention = GeometricAttentionLayer(
            mv_channels = hidden_dim,
            hidden_dim = hidden_dim,
            out_channels = hidden_dim,
            blade = blade,
            blade_len = blade_len,
            n_heads = n_heads
         )

        # Bilinear block
        self.bilinear_equinorm = EquilinearNormLayer()

        self.geometric_bilinear = GeometricBilinearLayer(
            in_mv_channels = hidden_dim,
            hidden_mv_channels = hidden_dim,
         )

        self.gated_gelu = GatedRELU()

        self.gelu_equilinear = EquiLinearLayer(
            input_mv_channels = hidden_dim,
            hidden_mv_dim = hidden_dim,
            blade = blade,
            blade_len = blade_len
         )

        # Final Equilinear Layer
        self.final_equilinear = EquiLinearLayer(
            input_mv_channels = hidden_dim,
            hidden_mv_dim = 256,
            blade = blade,
            blade_len = blade_len
         )

        # Vectorizer
        # self.vectorizer = nn.Linear(
        #     in_features=224 # fixed
        #     out_features=3
        # )

    def forward(self, x):
        # Entering Equilinear Layer
        projected_in_mv = self.enter_equilinear(x)

        # Attention block
        normalized_in_mv = self.attention_equinorm(
            projected_in_mv
         )

        attended_mv = self.geometric_attention(
            normalized_in_mv
         )

        # Residual connection 1
        residual_mv_1 = attended_mv + projected_in_mv

        # Bilinear block
        reference = torch.mean(x, dim = (1,2), keepdim = True)

        normalized_residual_mv = self.bilinear_equinorm(
            residual_mv_1
         )

        bilinear_mv = self.geometric_bilinear(
            normalized_residual_mv,
            reference
         )

        gated_mv = self.gated_gelu(bilinear_mv)

        projected_bilinear_mv = self.gelu_equilinear(
            gated_mv
         )

        # Residual connection 2
        residual_mv_2 = projected_bilinear_mv + residual_mv_1

        # Final Equilinear Layer
        output_mv = self.final_equilinear(residual_mv_2)
        return output_mv

        extracted_scalars = output_mv[:,:,0,:][...,[0]].squeeze(-1)
        print(f"Output scalars: {extracted_scalars.shape}")
        return extracted_scalars


    

class GATrToFoldingNetAdapter(nn.Module):
    def __init__(self, blade, blade_len, hidden_dim, intermediate_dim, output_dim, n_heads):
        super(GATrToFoldingNetAdapter, self).__init__()
        self.gatr_net = GATrNet(
            in_channels=16,  # Multivector input dimension
            blade=blade,
            blade_len=blade_len,
            hidden_dim=hidden_dim,
            n_heads=n_heads
        )
        # self.reduce_multivector = nn.Linear(16, intermediate_dim)  # Compress multivector dimension
        
        # Pooling layer to aggregate tokens
        self.token_pooling = nn.AdaptiveAvgPool2d((None, 1))  # Compress token dimension to 1
        
        self.project_to_folding = nn.Sequential(
            nn.Linear(intermediate_dim, output_dim),  # Compress final features to FoldingNet dimensions
            nn.ReLU()
        )

        self.reduce_dim = nn.Linear(
            in_features=16,
            out_features=128
        )

    def forward(self, point_cloud):
        # Step 1: Embed points into multivectors
        embedded_points = embed_point(point_cloud)  # Shape: (batch_size, 224, 16)
        embedded_points = embedded_points.unsqueeze(-2)  # Add channel dimension
        
        # Step 2: Pass through GaTr
        gatr_output = self.gatr_net(embedded_points)  # Shape: (batch_size, 224, 256, 16)
        return gatr_output

        reduced_mv = self.reduce_dim(gatr_output)  # [16, 224, 256, reduced_dim]
        reduced_mv = reduced_mv.mean(dim=2)

        return reduced_mv
        
        # Step 3: Reduce multivector dimension
        # reduced_multivectors = self.reduce_multivector(gatr_output)  # Shape: (batch_size, 224, 256, intermediate_dim)
        
        # Step 4: Pool across token dimension
        pooled_tokens = self.token_pooling(gatr_output).squeeze(-2).squeeze(-1) # Shape: (batch_size, 224, intermediate_dim)
        # print(pooled_tokens.shape)
        # exit()
        
        # Step 5: Project to FoldingNet input dimensions
        # print(pooled_tokens.squeeze().shape)
        # exit()
        folding_input = self.project_to_folding(pooled_tokens.squeeze())  # Shape: (batch_size, 224, output_dim)
        return folding_input


if __name__ == "__main__":

    blade = blade_operator().to('cuda')
    print(blade.shape[0])
    model = GATrToFoldingNetAdapter(
        blade=blade,
        blade_len=blade.shape[0],
        hidden_dim=254,  # Hidden dim for GATr
        intermediate_dim=256,  # fixed
        output_dim=3,  # Desired output dim for FoldingNet
        n_heads=4
    ).to('cuda')
    point_cloud = torch.rand((32, 224, 3), device='cuda')  # Example point cloud
    embedding = model(point_cloud)
    print(embedding.shape) 
