import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


class GAConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(GAConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Parameters for the rotation matrix
        self.phi = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.theta = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size) * 2 * torch.pi - torch.pi)
        self.pb = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.pc = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.pd = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))

    def rotation_matrix(self):
        """Compute the rotation matrix for the convolution kernel."""
        a = self.phi * torch.cos(self.theta)
        b = self.phi * self.pb * torch.sin(self.theta)
        c = self.phi * self.pc * torch.sin(self.theta)
        d = self.phi * self.pd * torch.sin(self.theta)
        
        # Construct the 3x3 rotation matrix as described
        rotation_matrix = torch.stack([
            torch.stack([1 - 2 * c**2 - 2 * d**2, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d], dim=-1),
            torch.stack([2 * b * c + 2 * a * d, 1 - 2 * b**2 - 2 * d**2, 2 * c * d - 2 * a * b], dim=-1),
            torch.stack([2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, 1 - 2 * b**2 - 2 * c**2], dim=-1)
        ], dim=-2)  # Shape: [out_channels, in_channels, kernel_size, 3, 3]
        
        return rotation_matrix

    def forward(self, x):
        # x shape: [batch_size, in_channels, spatial_dim]
        batch_size, in_channels, spatial_dim = x.size()
        
        # Compute rotation matrix
        rotation_matrix = self.rotation_matrix()  # Shape: [out_channels, in_channels, kernel_size, 3, 3]

        # Expand x for geometric product
        x = x.unsqueeze(2)  # Shape: [batch_size, in_channels, 1, spatial_dim]

        # Apply rotation matrix through einsum without summing over dimensions
        # Ensure the output shape is preserved as [batch_size, out_channels, spatial_dim]
        rotated_x = torch.einsum("bcis,ocijk->bojs", x, rotation_matrix)  # Shape: [batch_size, out_channels, 3, spatial_dim]

        # Aggregate across the rotation matrix's 3rd dimension if necessary for final output
        output = rotated_x.mean(dim=2)  # Shape: [batch_size, out_channels, spatial_dim]
        
        # print(f"GAConv1D output shape: {output.shape}")
        return output



class GAFold(nn.Module):
    def __init__(self, in_channel, step, hidden_dim=512):
        super().__init__()

        print(f"trans_dim: {in_channel}")
        print(f"step: {step}")
        print(f"hidden_dim: {hidden_dim}")

        self.in_channel = in_channel
        self.step = step

        # Folding seed for positional embedding
        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        # Update folding layers with GAConv1D and correct BatchNorm1d
        self.folding1 = nn.Sequential(
            # GAConv1D(in_channel + 2, hidden_dim, kernel_size=1),
            nn.Conv1d(in_channel + 2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),       # hidden_dim channels
            nn.ReLU(inplace=True),
            # GAConv1D(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim // 2),  # hidden_dim // 2 channels
            nn.ReLU(inplace=True),
            # GAConv1D(hidden_dim // 2, 3, kernel_size=1),  # Final output is 3 channels
            nn.Conv1d(hidden_dim // 2, 3, kernel_size=1),  # Final output is 3 channels
        )

        self.folding2 = nn.Sequential(
            GAConv1D(in_channel + 3, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),       # hidden_dim channels
            nn.ReLU(inplace=True),
            GAConv1D(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim // 2),  # hidden_dim // 2 channels
            nn.ReLU(inplace=True),
            GAConv1D(hidden_dim // 2, 3, kernel_size=1),  # Final output is 3 channels
        )

        self.cnn = nn.Conv1d(in_channel + 2, hidden_dim, 1)

    def forward(self, pointr_parameters):

        # Extract PoinTr parameters:
        coarse_point_cloud = pointr_parameters['coarse_point_cloud']
        x = pointr_parameters['rebuild_feature']
        xyz = pointr_parameters['xyz']
        B, M ,C = pointr_parameters['BMC']

        # Input shape: [batch_size * variable_dim, in_channel]
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        # Concatenate seed and features
        x = torch.cat([seed, features], dim=1)  # Shape [bs, in_channel + 2, num_sample]
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)  # Shape [bs, in_channel + 3, num_sample]
        fd2 = self.folding2(x)  # Shape [bs, 3, num_sample]

        # Reshape to output shape [batch_size * variable_dim, 384] if required
        folded_features = fd2.view(bs * num_sample, -1)

        # PoinTr steps
        relative_xyz = folded_features.reshape(B, M, 3, -1)    # B M 3 S
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3
        rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()

        return rebuild_points
    

    def fps(self, pc, num):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
        sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return sub_pc


if __name__ == "__main__":

    test = torch.ones([448, 386, 64], dtype=torch.float32, device='cuda')
    module = GAConv1D(
        in_channels=384,
        out_channels=512,
        kernel_size=1
    ).to('cuda')
    cnn = nn.Conv1d(384 + 2, 512, 1).to('cuda')
    output = cnn(test) #module(test)
    print(output.shape)