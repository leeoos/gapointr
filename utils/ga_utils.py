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
    