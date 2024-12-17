import torch

@torch.jit.script
def fast_einsum(q_einsum, cayley, k_einsum):
    """
    Implementation of the geometric product between two multivectors made with the einsum notation.
    Compiled with jit script for optimization!

    Args:
        q_einsum (torch.Tensor): left multivector
        cayley: look up tabel for the geometric product, it depends on the algebra used.
        k_einsum (torch.Tensor): right multivector.
    """
    return torch.einsum("...i,ijk,...k->...j", q_einsum, cayley, k_einsum)


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
        dim: int: starting dim, default: 0.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]
    

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