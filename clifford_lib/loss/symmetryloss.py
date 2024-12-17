import os 
import sys
import torch
import torch.nn as nn


# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../../'))
sys.path.append(os.path.join(BASE_DIR, '../../'))
sys.path.append(os.path.join(BASE_DIR, '../'))

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra
from utils.ga_utils import compute_volume_with_wedge
from utils.knn_utils import knn_gather, knn_points

# Chamfer distance 
from extensions.chamfer_dist import ChamferDistanceL1

class SymmetryLoss(nn.Module):
    def __init__(self, lambda_rot=1.0, lambda_ref=1.0, lambda_align=1.0, lambda_geo=1.0):
        """
        Initialize the Symmetry Preserving Loss Function.
        
        Args:
            lambda_rot: Weight for rotational symmetry loss.
            lambda_ref: Weight for reflectional symmetry loss.
            lambda_align: Weight for alignment loss (e.g., Chamfer distance).
            lambda_geo: Weight for geometric consistency loss.
        """
        super(SymmetryLoss, self).__init__()
        self.lambda_rot = lambda_rot
        self.lambda_ref = lambda_ref
        self.lambda_align = lambda_align
        self.lambda_geo = lambda_geo

    def compute_rotational_symmetry_loss(self, source_points, target_points, axis):
        """
        Compute rotational symmetry loss by comparing rotational invariants.
        
        Args:
            source_points: Tensor of input points, shape (B, N, 3).
            target_points: Tensor of reconstructed points, shape (B, N, 3).
            axis: Tensor representing the axis of rotation, shape (B, 3).
        
        Returns:
            Rotational symmetry loss.
        """
        # Project points onto the rotational axis
        input_proj = torch.einsum('bij,bj->bi', source_points, axis)  # Dot product with axis
        recon_proj = torch.einsum('bij,bj->bi', target_points, axis)
        
        # Compute rotational invariants (magnitudes)
        input_magnitude = torch.norm(input_proj, dim=-1)
        recon_magnitude = torch.norm(recon_proj, dim=-1)
        
        # Compute loss
        loss_rot = torch.mean((input_magnitude - recon_magnitude) ** 2)
        return loss_rot

    def compute_reflectional_symmetry_loss(self, source_points, target_points, plane_normal):
        """
        Compute reflectional symmetry loss by comparing distances to the symmetry plane.
        
        Args:
            source_points: Tensor of input points, shape (B, N, 3).
            target_points: Tensor of reconstructed points, shape (B, N, 3).
            plane_normal: Tensor representing the normal to the symmetry plane, shape (B, 3).
        
        Returns:
            Reflectional symmetry loss.
        """
        # Normalize the plane normal
        plane_normal = plane_normal / torch.norm(plane_normal, dim=-1, keepdim=True)
        
        # Compute distances from points to the plane
        input_dist = torch.einsum('bij,bj->bi', source_points, plane_normal)
        recon_dist = torch.einsum('bij,bj->bi', target_points, plane_normal)
        
        # Compute loss
        loss_ref = torch.mean((input_dist - recon_dist) ** 2)
        return loss_ref

    def compute_alignment_loss(self, source_points, target_points):
        """
        Compute alignment loss using Chamfer Distance.
        
        Args:
            source_points: Tensor of input points, shape (B, N, 3).
            target_points: Tensor of reconstructed points, shape (B, M, 3).
        
        Returns:
            Alignment loss (Chamfer Distance).
        """
        # Compute pairwise distances
        # input_distances = torch.cdist(source_points, target_points, p=2)  # Shape (B, N, M)
        
        # Chamfer distance
        # min_dist_input = torch.min(input_distances, dim=-1)[0]  # Minimum for each input point
        # min_dist_recon = torch.min(input_distances, dim=-2)[0]  # Minimum for each recon point
        # loss_align = torch.mean(min_dist_input) + torch.mean(min_dist_recon)
        # return loss_align
    
        # or 
        cdl1 = ChamferDistanceL1()
        return cdl1(source_points, target_points)
    
    def compute_geometric_consistency_loss(self, source_points, target_points):
        """
        Compute geometric consistency loss using the wedge product for volume-based measures.

        Args:
            source_points: Tensor of input points, shape (B, N, 3).
            target_points: Tensor of reconstructed points, shape (B, N, 3).

        Returns:
            Geometric consistency loss.
        """
        input_volume = compute_volume_with_wedge(source_points)
        recon_volume = compute_volume_with_wedge(target_points)

        # Loss is the difference in volumes
        loss_geo = torch.mean((input_volume - recon_volume) ** 2)
        return loss_geo



    def forward(self, source_points: torch.Tensor, target_points: torch.Tensor, device='cuda'):
        """
        Compute the total symmetry-preserving loss.
        
        Args:
            source_points: Tensor of input points, shape (B, N, 3).
            target_points: Tensor of reconstructed points, shape (B, N, 3).
            axis: Tensor representing the axis of rotation, shape (B, 3).
            plane_normal: Tensor representing the normal to the symmetry plane, shape (B, 3).
        
        Returns:
            Total symmetry-preserving loss.
        """

        # Initialize plane and reference axis
        batchsize_source, lengths_source, dim_source = source_points[1].shape
        batchsize_target, lengths_target, dim_target = target_points.shape
        axis = torch.tensor([[0., 0., 1.]], requires_grad=True).repeat(batchsize_source, 1).to(device)
        plane_normal = torch.tensor([[0., 1., 0.]], requires_grad=True).repeat(batchsize_source, 1).to(device)


        # KNN computation
        lengths_source = (
            torch.ones(batchsize_source, dtype=torch.long, device=source_points[1].device)
            * lengths_source
        )
        lengths_target = (
            torch.ones(batchsize_target, dtype=torch.long, device=target_points.device)
            * lengths_target
        )
        source_nn = knn_points(
            source_points[1],
            target_points,
            lengths1=lengths_source,
            lengths2=lengths_target,
            K=1,
        )
        new_target_points = knn_gather(target_points, idx=source_nn.idx, lengths=lengths_target).squeeze(-2)
        

        # Compute individual losses
        loss_align_coarse = self.compute_alignment_loss(source_points[0], target_points)
        loss_align_fine = self.compute_alignment_loss(source_points[1], target_points)
        loss_rot = self.compute_rotational_symmetry_loss(source_points[1], new_target_points, axis)
        loss_ref = self.compute_reflectional_symmetry_loss(source_points[1], new_target_points, plane_normal)
        loss_geo = self.compute_geometric_consistency_loss(source_points[1], new_target_points)
        
        # Combine losses
        total_loss = (
            self.lambda_rot * loss_rot +
            self.lambda_ref * loss_ref +
            self.lambda_align * loss_align_coarse +
            self.lambda_align * loss_align_fine +
            self.lambda_geo * loss_geo
        )
        return total_loss



if __name__ == "__main__":
    loss_fn = SymmetryLoss(lambda_rot=1.0, lambda_ref=1.0, lambda_align=1.0, lambda_geo=1.0)

    batch_size = 32
    num_points = 2048

    # Example tensors
    source_points = torch.randn((batch_size, num_points, 3))  # Input partial point cloud
    target_points = torch.randn((batch_size, num_points, 3))  # Reconstructed point cloud
    axis = torch.tensor([[0., 0., 1.]], requires_grad=True).repeat(batch_size, 1)  # Rotational axis (e.g., z-axis)
    plane_normal = torch.tensor([[0., 1., 0.]], requires_grad=True).repeat(batch_size, 1)  # Reflection plane normal (e.g., y-plane)

    # Compute loss
    loss = loss_fn(source_points, target_points, axis, plane_normal)
    print(f"loss: {loss.item()}")
    print("Attempting backward pass!")
    loss.backward()
    print("done!")
