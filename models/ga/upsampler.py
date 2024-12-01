import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss

# Chamfer Distance Loss for geometric consistency
class ChamferDistanceLoss(nn.Module):
    def forward(self, pred, gt):
        """
        :param pred: Predicted point cloud (batch_size, n_points, 3)
        :param gt: Ground truth point cloud (batch_size, n_points, 3)
        :return: Chamfer distance between pred and gt
        """
        pred_expand = pred.unsqueeze(2)  # (batch_size, pred_points, 1, 3)
        gt_expand = gt.unsqueeze(1)      # (batch_size, 1, gt_points, 3)

        # Squared distances
        distances = torch.sum((pred_expand - gt_expand) ** 2, dim=-1)  # (batch_size, pred_points, gt_points)

        # Nearest neighbor distances
        pred_to_gt = torch.min(distances, dim=2)[0]
        gt_to_pred = torch.min(distances, dim=1)[0]

        # Chamfer distance
        chamfer_distance = torch.mean(pred_to_gt) + torch.mean(gt_to_pred)
        return chamfer_distance


class PointCloudUpsamplerImproved(nn.Module):
    def __init__(self, input_points=448, output_points=2048):
        super(PointCloudUpsamplerImproved, self).__init__()
        self.input_points = input_points
        self.output_points = output_points

        # Calculate expansion factor and remaining points
        self.expansion_factor = output_points // input_points
        self.remaining_points = output_points - (input_points * self.expansion_factor)

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Local context propagation
        self.context_layer = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.ReLU()
        )

        # Expansion layer
        self.expansion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.expansion_factor * 3)
        )

        # Regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        :param x: Input point cloud of shape (batch_size, input_points, 3)
        :return: Upsampled point cloud of shape (batch_size, output_points, 3)
        """
        batch_size, num_points, _ = x.size()
        assert num_points == self.input_points, f"Expected {self.input_points} points, got {num_points}"

        # Feature extraction
        features = self.feature_extractor(x)  # (batch_size, input_points, 256)

        # Local context propagation
        features = features.permute(0, 2, 1)  # Convert to (batch_size, channels, points)
        context_features = self.context_layer(features)  # (batch_size, 1024, input_points)
        context_features = context_features.permute(0, 2, 1)  # Convert back to (batch_size, input_points, 1024)

        # Expand points
        expanded = self.expansion(context_features)  # (batch_size, input_points, expansion_factor * 3)
        expanded = self.dropout(expanded)

        # Reshape to initial expanded dimensions
        expanded = expanded.view(batch_size, self.input_points * self.expansion_factor, 3)

        # Handle remaining points
        if self.remaining_points > 0:
            additional_points = expanded[:, :self.remaining_points, :]  # Use the first few expanded points
            expanded = torch.cat([expanded, additional_points], dim=1)

        return expanded



# Custom Loss Function
class UpsamplingLoss(nn.Module):
    def __init__(self):
        super(UpsamplingLoss, self).__init__()
        self.chamfer_loss = ChamferDistanceLoss()
        self.smoothness_loss = MSELoss()

    def forward(self, pred, gt):
        """
        :param pred: Predicted point cloud (batch_size, n_points, 3)
        :param gt: Ground truth point cloud (batch_size, n_points, 3)
        :return: Combined loss
        """
        chamfer = self.chamfer_loss(pred, gt)
        smoothness = self.smoothness_loss(pred[:, 1:, :], pred[:, :-1, :])  # Penalize large jumps in adjacent points
        return chamfer + 0.1 * smoothness  # Weighted sum


# Example usage
if __name__ == "__main__":
    batch_size = 8
    input_points = 448
    output_points = 2048

    # Create random input point cloud
    input_cloud = torch.randn(batch_size, input_points, 3)
    ground_truth = torch.randn(batch_size, output_points, 3)

    # Create upsampler model
    upsampler = PointCloudUpsamplerImproved(input_points=input_points, output_points=output_points)

    # Perform upsampling
    upsampled_cloud = upsampler(input_cloud)

    # Define loss function
    criterion = UpsamplingLoss()
    loss = criterion(upsampled_cloud, ground_truth)

    print(f"Input shape: {input_cloud.shape}")
    print(f"Upsampled shape: {upsampled_cloud.shape}")
    print(f"Loss: {loss.item()}")
