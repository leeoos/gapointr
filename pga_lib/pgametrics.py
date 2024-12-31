import torch

def compute_volume_with_wedge(points):
    """
    Compute volume using the wedge product (generalized cross-product).

    Args:
        points: Tensor of shape (B, N, 3), where B is batch size, N is number of points, 3 is for x, y, z.

    Returns:
        Volumes for each batch, averaged across all points.
    """
    # Extract triplets of points to compute wedge product
    p1 = points[:, :-2, :]  # Shape (B, N-2, 3)
    p2 = points[:, 1:-1, :]
    p3 = points[:, 2:, :]

    # Compute wedge product: p1 ∧ p2 ∧ p3 (volume of parallelepiped)
    # Equivalent to computing determinant of a 3x3 matrix formed by these vectors
    wedge_products = torch.cross(p2 - p1, p3 - p1, dim=-1)  # Cross product of two vectors
    volumes = torch.abs(torch.sum(wedge_products * (p3 - p2), dim=-1))  # Compute oriented volume

    # Average across batches and points
    return torch.mean(volumes, dim=-1)  # Shape (B,)