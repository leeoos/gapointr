# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""Functions that embed points in the geometric algebra."""



import torch

# from gatr.utils.warning import GATrDeprecationWarning


def embed_point(coordinates: torch.Tensor) -> torch.Tensor:
    """Embeds 3D points in multivectors.

    We follow the convention used in the reference below and map points to tri-vectors.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
    https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    coordinates : torch.Tensor with shape (..., 3)
        3D coordinates

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = coordinates.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=coordinates.dtype, device=coordinates.device)

    # Embedding into trivectors
    # Homogeneous coordinates: unphysical component / embedding dim, x_123
    multivector[..., 14] = 1.0
    multivector[..., 13] = -coordinates[..., 0]  # x-coordinate embedded in x_023
    multivector[..., 12] = coordinates[..., 1]  # y-coordinate embedded in x_013
    multivector[..., 11] = -coordinates[..., 2]  # z-coordinate embedded in x_012

    return multivector


def extract_point(
    multivector: torch.Tensor, divide_by_embedding_dim: bool = True, threshold: float = 1e-3
) -> torch.Tensor:
    """Given a multivector, extract any potential 3D point from the trivector components.

    Nota bene: if the output is interpreted a regular R^3 point,
    this function is only equivariant if divide_by_embedding_dim=True
    (or if the e_123 component is guaranteed to equal 1)!

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.
    divide_by_embedding_dim : bool
        Whether to divice by the embedding dim. Proper PGA etiquette would have us do this, but it
        may not be good for NN training. If set the False, this function is not equivariant for all
        inputs!
    threshold : float
        Minimum value of the additional, unphysical component. Necessary to avoid exploding values
        or NaNs when this unphysical component of the homogeneous coordinates becomes small.

    Returns
    -------
    coordinates : torch.Tensor with shape (..., 3)
        3D coordinates corresponding to the trivector components of the multivector.
    """
    # if not divide_by_embedding_dim:
    #     warnings.warn(
    #         'Calling "extract_point" with divide_by_embedding_dim=False is deprecated, '
    #         "because it is not equivariant.",
    #         GATrDeprecationWarning,
    #         2,
    #     )

    coordinates = torch.cat(
        [-multivector[..., [13]], multivector[..., [12]], -multivector[..., [11]]], dim=-1
    )

    # Divide by embedding dim
    if divide_by_embedding_dim:
        embedding_dim = multivector[
            ..., [14]
        ]  # Embedding dimension / scale of homogeneous coordinates
        embedding_dim = torch.where(torch.abs(embedding_dim) > threshold, embedding_dim, threshold)
        coordinates = coordinates / embedding_dim

    return coordinates


def extract_point_embedding_reg(multivector: torch.Tensor) -> torch.Tensor:
    """Given a multivector x, returns |x_{123}| - 1.

    Put differently, this is the deviation of the norm of a pseudoscalar component from 1.
    This can be used as a regularization term when predicting point positions, to avoid x_123 to
    be too close to 0.

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.

    Returns
    -------
    regularization : torch.Tensor with shape (..., 1)
        |multivector_123| - 1.
    """

    return torch.abs(multivector[..., [14]]) - 1.0


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from pathlib import Path

# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""This module provides efficiency improvements over torch's einsum through caching."""

import functools
from typing import Any, Callable, List, Sequence

import opt_einsum
import torch


def _einsum_with_path(equation: str, *operands: torch.Tensor, path: List[int]) -> torch.Tensor:
    """Computes einsum with a given contraction path."""

    # Justification: For the sake of performance, we need direct access to torch's private methods.

    # pylint:disable-next=protected-access
    return torch._VF.einsum(equation, operands, path=path)  # type: ignore[attr-defined]


def _einsum_with_path_ignored(equation: str, *operands: torch.Tensor, **kwargs: Any):
    """Calls torch.einsum whilst dropping all kwargs.

    Allows use of hard-coded optimal contraction paths in `gatr_einsum_with_path` for
    non-compiling code whilst dropping the optimal contraction path for compiling code.
    """
    return torch.einsum(equation, *operands)


def _cached_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Computes einsum whilst caching the optimal contraction path.

    Inspired by upstream
    https://github.com/pytorch/pytorch/blob/v1.13.0/torch/functional.py#L381.
    """
    op_shape = tuple(op.shape for op in operands)
    path = _get_cached_path_for_equation_and_shapes(equation=equation, op_shape=op_shape)

    return _einsum_with_path(equation, *operands, path=path)


@functools.lru_cache(maxsize=None)
def _get_cached_path_for_equation_and_shapes(
    equation: str, op_shape: Sequence[torch.Tensor]
) -> List[int]:
    """Provides shape-based caching of the optimal contraction path."""
    tupled_path = opt_einsum.contract_path(equation, *op_shape, optimize="optimal", shapes=True)[0]

    return [item for pair in tupled_path for item in pair]


class gatr_cache(dict):
    """Serves as a `torch.compile`-compatible replacement for `@functools.cache()`."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def __missing__(self, item: Any) -> Any:
        """Computes missing function values and adds them to the cache."""
        tensor = self.fn(*item)
        self[item] = tensor
        return tensor

    def __call__(self, *args: Any) -> Any:
        """Allows to access cached function values with `()` instead of `[]`."""
        return self[args]


_gatr_einsum = _cached_einsum
_gatr_einsum_with_path = _einsum_with_path


def gatr_einsum(equation: str, *operands: torch.Tensor):
    """Computes torch.einsum with contraction path caching if enabled (and compilation is not used).

    Cf. `enable_cached_einsum` for more context.
    """
    return _gatr_einsum(equation, *operands)


def gatr_einsum_with_path(equation: str, *operands: torch.Tensor, path: List[int]):
    """Computes einsum with a given contraction path (which is ignored when using compilation).

    Cf. `enable_cached_einsum` for more context.
    """
    return _gatr_einsum_with_path(equation, *operands, path=path)


def enable_cached_einsum(flag: bool) -> None:
    """Selects whether to use caching of optimal paths in einsum contraction computations.

    When using torch.compile (torch==2.2.1), if we specify the precomputed paths when calling
    `torch._VF.einsum(equation, operands, path=path)`, the compiler errors out.

    Thus, users who wish to use `torch.compile` need to disable caching of einsum
    by calling `enable_cached_einsum(False)`.

    By default, caching is used, as we currently expect less users to use compilation.
    """
    global _gatr_einsum
    global _gatr_einsum_with_path
    if flag:
        _gatr_einsum = _cached_einsum
        _gatr_einsum_with_path = _einsum_with_path
    else:
        _gatr_einsum = torch.einsum
        _gatr_einsum_with_path = _einsum_with_path_ignored


_FILENAMES = {"gp": "geometric_product.pt", "outer": "outer_product.pt"}


@gatr_cache
def _load_bilinear_basis(
    kind: str, device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Loads basis elements for Pin-equivariant bilinear maps between multivectors.

    Parameters
    ----------
    kind : {"gp", "outer"}
        Filename of the basis file, assumed to be found in __file__ / data
    device : torch.Device or str
        Device
    dtype : torch.Dtype
        Data type

    Returns
    -------
    basis : torch.Tensor with shape (num_basis_elements, 16, 16, 16)
        Basis elements for bilinear equivariant maps between multivectors.
    """

    # To avoid duplicate loading, base everything on float32 CPU version
    if device not in [torch.device("cpu"), "cpu"] and dtype != torch.float32:
        basis = _load_bilinear_basis(kind)
    else:
        filename = Path(__file__).parent.resolve() / "data" / _FILENAMES[kind]
        sparse_basis = torch.load(filename).to(torch.float32)
        # Convert to dense tensor
        # The reason we do that is that einsum is not defined for sparse tensors
        basis = sparse_basis.to_dense()

    return basis.to(device=device, dtype=dtype)


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the geometric product f(x,y) = xy.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    gp = _load_bilinear_basis("gp", x.device, x.dtype)

    # Compute geometric product
    outputs = gatr_einsum("i j k, ... j, ... k -> ... i", gp, x, y)

    return outputs


def outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the outer product `f(x,y) = x ^ y`.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    op = _load_bilinear_basis("outer", x.device, x.dtype)

    # Compute geometric product
    outputs = gatr_einsum("i j k, ... j, ... k -> ... i", op, x, y)

    return outputs



if __name__ == "__main__":

    pcn_input_1 = torch.rand((32, 2048, 3))
    pcn_input_2 = torch.rand((32, 2048, 3))
    multivector_1 = embed_point(pcn_input_1)
    multivector_2 = embed_point(pcn_input_2)
    result= geometric_product(multivector_1, multivector_2)
    print(f"multivector shape: {result.shape}")
