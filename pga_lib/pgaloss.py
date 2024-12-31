import os 
import sys
import torch
import torch.nn as nn


# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../'))

# KNN
from utils.knn_utils import knn_gather, knn_points

# Chamfer distance 
from extensions.chamfer_dist import ChamferDistanceL1

# PGA
from pga import *

class PGALoss(nn.Module):
    def __init__(self):
        """
        Initialize the Symmetry Preserving Loss Function.
        
        Args:
            ....
        """
        super(PGALoss, self).__init__()

    def chamfer_distance_loss(self, predicted, target):
        chdl1 = ChamferDistanceL1()
        return chdl1(predicted, target)

    def sandwitch_loss(self, predicted, target, u, inv_u):
        transformed_predicted = sandwich_product(predicted, u, inv_u)
        return torch.norm(transformed_predicted - target, dim=-1).mean()


    def join_loss(self, predicted, target):
        # Compute reference multivector as the mean of both point clouds
        reference = (predicted.mean(dim=1, keepdim=True) + target.mean(dim=1, keepdim=True)) / 2  # Shape: (batch_size, 1, 16)

        # Compute join operation
        join_result = join(target, target, reference)  # Shape: (batch_size, 2048, 16)
        # print(join_result)

        # Measure difference using norm of the join result
        join_norm = torch.norm(join_result, dim=-1)  # Shape: (batch_size, 2048)
        # print(join_norm)

        # Average over all points in the batch
        loss = torch.mean(join_norm)  # Scalar

        return loss
    

    def geometric_product_loss(self, prediction, target):
      
        # Compute geometric product between corresponding points
        geom_prod = geometric_product(prediction, target)  # Shape: (batch_size, 2048, 16)

        # Compute the difference between embedded multivectors
        mv_diff = prediction - target  # Shape: (batch_size, 2048, 16)

        # Apply dual operation to the difference
        dual_flip, dual_signs = compute_dualization() #dual_operators()
        dual_mv_diff = dual_signs * mv_diff[..., dual_flip]

        # Measure the norm of the dualized difference
        dual_norm = torch.norm(dual_mv_diff, dim=-1)  # Shape: (batch_size, 2048)

        # Combine norms to create the loss
        loss = torch.mean(torch.norm(geom_prod, dim=-1) + dual_norm)  # Scalar

        # Empirically determine min and max values
        min_loss = 1.0  # Adjust based on the observed minimum in perfect matches
        max_loss = 3.0  # Example maximum, adjust based on observed values

        # Normalize between 0 and 1
        normalized_loss = (loss - min_loss) / (max_loss - min_loss)
        normalized_loss = torch.clamp(normalized_loss, 0.0, 1.0)  # Ensure within bounds

        return normalized_loss


    def forward(self, source_points: torch.Tensor, target_points: torch.Tensor, device='cuda'):
        """
        Compute the total symmetry-preserving loss.
        
        Args:
            source_points: Tensor of input points, shape (B, N, 3).
            target_points: Tensor of reconstructed points, shape (B, N, 3).
        
        Returns:
            Total symmetry-preserving loss.
        """

        # Initialize plane and reference axis
        batchsize_source, lengths_source, dim_source = source_points.shape
        batchsize_target, lengths_target, dim_target = target_points.shape

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
            source_points,
            target_points,
            lengths1=lengths_source,
            lengths2=lengths_target,
            K=1,
        )
        new_target_points = knn_gather(target_points, idx=source_nn.idx, lengths=lengths_target).squeeze(-2)

        target_nn = knn_points(
            target_points,
            source_points,
            lengths1=lengths_target,
            lengths2=lengths_source,
            K=1,
        )
        new_source_points = knn_gather(source_points, idx=target_nn.idx, lengths=lengths_target).squeeze(-2)


        # MV embedding
        # MV embedding
        mv_source = embed_point(source_points)
        mv_target = embed_point(target_points)

        mv_new_source = embed_point(new_source_points)
        mv_new_target = embed_point(new_target_points)

        # j_loss = self.join_loss(mv_source, mv_target)
        gp_loss_t2s = self.geometric_product_loss(mv_source, mv_new_target)
        gp_loss_s2t = self.geometric_product_loss(mv_new_source, mv_target)

        # Combine losses
        total_loss = 0.5*(gp_loss_t2s + gp_loss_s2t)
        return total_loss



if __name__ == "__main__":
    loss_fn = PGALoss(lambda_rot=1.0, lambda_ref=1.0, lambda_align=1.0, lambda_geo=1.0)

    batch_size = 32
    num_points = 2048

    # Example tensors
    source_points_1 = torch.randn((batch_size, num_points, 3)).to('cuda')  # Input partial point cloud
    source_points_2 = torch.randn((batch_size, num_points, 3)).to('cuda')  # Input partial point cloud
    target_points = torch.randn((batch_size, num_points, 3)).to('cuda')  # Reconstructed point cloud
    axis = torch.tensor([[0., 0., 1.]], requires_grad=True).repeat(batch_size, 1)  # Rotational axis (e.g., z-axis)
    plane_normal = torch.tensor([[0., 1., 0.]], requires_grad=True).repeat(batch_size, 1)  # Reflection plane normal (e.g., y-plane)

    # Compute loss
    loss = loss_fn((source_points_1, source_points_2) , target_points)
    print(f"loss: {loss.item()}")
    print("Attempting backward pass!")
    loss.backward()
    print("done!")
