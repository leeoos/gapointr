import os 
import sys


# Setup base directory and add file to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

# Clifford Algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra


# Stolen from pytorch3d, and tweaked (very little).

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import warnings
from collections import namedtuple
from typing import Optional, Union

import torch

# Throws an error without this import
from chamferdist import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable


_KNN = namedtuple("KNN", "dists idx knn")

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


class MVLoss(torch.nn.Module):
    def __init__(self, metric):
        super(MVLoss, self).__init__()
        self.ca = CliffordAlgebra(metric)

    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
        bidirectional: Optional[bool] = False,
        reverse: Optional[bool] = False,
        batch_reduction: Optional[str] = "mean",
        point_reduction: Optional[str] = "sum",
    ):

        if not isinstance(source_cloud, torch.Tensor):
            raise TypeError(
                "Expected input type torch.Tensor. Got {} instead".format(type(source_cloud))
            )
        if not isinstance(target_cloud, torch.Tensor):
            raise TypeError(
                "Expected input type torch.Tensor. Got {} instead".format(type(target_cloud))
            )
        if source_cloud.device != target_cloud.device:
            raise ValueError(
                "Source and target clouds must be on the same device. "
                f"Got {source_cloud.device} and {target_cloud.device}."
            )

        batchsize_source, lengths_source, dim_source = source_cloud.shape
        batchsize_target, lengths_target, dim_target = target_cloud.shape

        lengths_source = (
            torch.ones(batchsize_source, dtype=torch.long, device=source_cloud.device)
            * lengths_source
        )
        lengths_target = (
            torch.ones(batchsize_target, dtype=torch.long, device=target_cloud.device)
            * lengths_target
        )

        chamfer_dist = None

        if batchsize_source != batchsize_target:
            raise ValueError(
                "Source and target pointclouds must have the same batchsize."
            )
        if dim_source != dim_target:
            raise ValueError(
                "Source and target pointclouds must have the same dimensionality."
            )
        if bidirectional and reverse:
            warnings.warn(
                "Both bidirectional and reverse set to True. "
                "bidirectional behavior takes precedence."
            )
        if point_reduction != "sum" and point_reduction != "mean" and point_reduction != None:
            raise ValueError('Point reduction must either be "sum" or "mean" or None.')
        if batch_reduction != "sum" and batch_reduction != "mean" and batch_reduction != None:
            raise ValueError('Batch reduction must either be "sum" or "mean" or None.')

        source_nn = knn_points(
            source_cloud,
            target_cloud,
            lengths1=lengths_source,
            lengths2=lengths_target,
            K=1,
        )

        # print(source_nn.idx)
        # print(source_nn.idx.shape)
        p_nn = knn_gather(target_cloud, idx=source_nn.idx, lengths=lengths_target).squeeze(-2)
        # print(p_nn)
        # print(p_nn.shape)
        # mv computation
        cayley = self.ca.cayley.to('cuda')
        mv_output = self.ca.embed_grade(source_cloud, 1)
        mv_target = self.ca.embed_grade(p_nn, 1)

        # print(mv_output.shape)
        # print(mv_target.shape)

        # Memory optimization
        # Make tensor contigous in memory for performance optimization
        mv_output = mv_output.contiguous()
        mv_target = mv_target.contiguous()
        cayley = cayley.contiguous()

        # Half precision for performance optimization
        mv_output = mv_output.half()
        mv_target = mv_target.half()
        cayley = cayley.half()

        mv_output_matrix = fast_einsum(mv_output.unsqueeze(1), cayley, mv_output.unsqueeze(2))
        mv_target_matrix = fast_einsum(mv_target.unsqueeze(1), cayley, mv_target.unsqueeze(2))

        # print(f"Output multivectors pairs: {mv_output_matrix}\n")
        # print(f"Target multivectors pairs: {mv_target_matrix}\n")

        # Ensure tensors are of the same shape
        # assert mv_target_matrix.shape == mv_output_matrix.shape, "Input tensors must have the same shape."
        
        # # Compute the difference between the tensors
        # diff = mv_target_matrix - mv_output_matrix  # Shape: (b, n, n, k)
        
        # # Square the differences and sum along the last dimension (k)
        # squared_diff = diff ** 2  # Shape: (b, n, n, k)
        # summed_diff = torch.sum(squared_diff, dim=-1)  # Shape: (b, n, n)
        
        # # Take the square root to get the L2 distance
        # distances = torch.sqrt(summed_diff)  # Shape: (b, n, n)

        # # print(distances)
        # # print(distances.shape)

        # mean_distances = torch.mean(distances, dim=-1)
        # torch.mean(mean_distances, dim=-1)

        # Compute the squared difference
        squared_difference = (mv_output_matrix - mv_target_matrix) ** 2

        # Compute the mean over all elements
        mse_loss = torch.mean(squared_difference)
        # mse_loss = squared_difference.sum() / squared_difference.numel()

        # print(mean_distances.shape)
        return mse_loss


        # print(f"loss: {mean_distances}")
        # print(mean_distances.shape)

        # target_nn = None
        # if reverse or bidirectional:
        #     target_nn = knn_points(
        #         target_cloud,
        #         source_cloud,
        #         lengths1=lengths_target,
        #         lengths2=lengths_source,
        #         K=1,
        #     )

        # # Forward Chamfer distance (batchsize_source, lengths_source)
        # chamfer_forward = source_nn.dists[..., 0]
        # chamfer_backward = None
        # if reverse or bidirectional:
        #     # Backward Chamfer distance (batchsize_source, lengths_source)
        #     chamfer_backward = target_nn.dists[..., 0]

        # if point_reduction == "sum":
        #     chamfer_forward = chamfer_forward.sum(1)  # (batchsize_source,)
        #     if reverse or bidirectional:
        #         chamfer_backward = chamfer_backward.sum(1)  # (batchsize_target,)
        # elif point_reduction == "mean":
        #     chamfer_forward = chamfer_forward.mean(1)  # (batchsize_source,)
        #     if reverse or bidirectional:
        #         chamfer_backward = chamfer_backward.mean(1)  # (batchsize_target,)

        # if batch_reduction == "sum":
        #     chamfer_forward = chamfer_forward.sum()  # (1,)
        #     if reverse or bidirectional:
        #         chamfer_backward = chamfer_backward.sum()  # (1,)
        # elif batch_reduction == "mean":
        #     chamfer_forward = chamfer_forward.mean()  # (1,)
        #     if reverse or bidirectional:
        #         chamfer_backward = chamfer_backward.mean()  # (1,)

        # if bidirectional:
        #     return chamfer_forward + chamfer_backward
        # if reverse:
        #     return chamfer_backward

        # return chamfer_forward


class _knn_points(Function):
    """
    Torch autograd Function wrapper for KNN C++/CUDA implementations.
    """

    @staticmethod
    def forward(
        ctx, p1, p2, lengths1, lengths2, K, version, return_sorted: bool = True
    ):
        """
        K-Nearest neighbors on point clouds.
        Args:
            p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
                containing up to P1 points of dimension D.
            p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
                containing up to P2 points of dimension D.
            lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
                length of each pointcloud in p1. Or None to indicate that every cloud has
                length P1.
            lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
                length of each pointcloud in p2. Or None to indicate that every cloud has
                length P2.
            K: Integer giving the number of nearest neighbors to return.
            version: Which KNN implementation to use in the backend. If version=-1,
                the correct implementation is selected based on the shapes of the inputs.
            return_sorted: (bool) whether to return the nearest neighbors sorted in
                ascending order of distance.
        Returns:
            p1_dists: Tensor of shape (N, P1, K) giving the squared distances to
                the nearest neighbors. This is padded with zeros both where a cloud in p2
                has fewer than K points and where a cloud in p1 has fewer than P1 points.
            p1_idx: LongTensor of shape (N, P1, K) giving the indices of the
                K nearest neighbors from points in p1 to points in p2.
                Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
                neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
                in p2 has fewer than K points and where a cloud in p1 has fewer than P1 points.
        """

        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        idx, dists = _C.knn_points_idx(p1, p2, lengths1, lengths2, K, version)

        # sort KNN in ascending order if K > 1
        if K > 1 and return_sorted:
            if lengths2.min() < K:
                P1 = p1.shape[1]
                mask = lengths2[:, None] <= torch.arange(K, device=dists.device)[None]
                # mask has shape [N, K], true where dists irrelevant
                mask = mask[:, None].expand(-1, P1, -1)
                # mask has shape [N, P1, K], true where dists irrelevant
                dists[mask] = float("inf")
                dists, sort_idx = dists.sort(dim=2)
                dists[mask] = 0
            else:
                dists, sort_idx = dists.sort(dim=2)
            idx = idx.gather(2, sort_idx)

        ctx.save_for_backward(p1, p2, lengths1, lengths2, idx)
        ctx.mark_non_differentiable(idx)
        return dists, idx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idx):
        p1, p2, lengths1, lengths2, idx = ctx.saved_tensors
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        if not (grad_dists.dtype == torch.float32):
            grad_dists = grad_dists.float()
        if not (p1.dtype == torch.float32):
            p1 = p1.float()
        if not (p2.dtype == torch.float32):
            p2 = p2.float()
        grad_p1, grad_p2 = _C.knn_points_backward(
            p1, p2, lengths1, lengths2, idx, grad_dists
        )
        return grad_p1, grad_p2, None, None, None, None, None


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
):
    """
    K-Nearest neighbors on point clouds.
    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
            containing up to P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
            containing up to P2 points of dimension D.
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
            length of each pointcloud in p1. Or None to indicate that every cloud has
            length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
            length of each pointcloud in p2. Or None to indicate that every cloud has
            length P2.
        K: Integer giving the number of nearest neighbors to return.
        version: Which KNN implementation to use in the backend. If version=-1,
            the correct implementation is selected based on the shapes of the inputs.
        return_nn: If set to True returns the K nearest neighbors in p2 for each point in p1.
        return_sorted: (bool) whether to return the nearest neighbors sorted in
            ascending order of distance.
    Returns:
        dists: Tensor of shape (N, P1, K) giving the squared distances to
            the nearest neighbors. This is padded with zeros both where a cloud in p2
            has fewer than K points and where a cloud in p1 has fewer than P1 points.
        idx: LongTensor of shape (N, P1, K) giving the indices of the
            K nearest neighbors from points in p1 to points in p2.
            Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
            neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
            in p2 has fewer than K points and where a cloud in p1 has fewer than P1
            points.
        nn: Tensor of shape (N, P1, K, D) giving the K nearest neighbors in p2 for
            each point in p1. Concretely, `p2_nn[n, i, k]` gives the k-th nearest neighbor
            for `p1[n, i]`. Returned if `return_nn` is True.
            The nearest neighbors are collected using `knn_gather`
            .. code-block::
                p2_nn = knn_gather(p2, p1_idx, lengths2)
            which is a helper function that allows indexing any tensor of shape (N, P2, U) with
            the indices `p1_idx` returned by `knn_points`. The outout is a tensor
            of shape (N, P1, K, U).
    """
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("pts1 and pts2 must have the same point dimension.")

    p1 = p1.contiguous()
    p2 = p2.contiguous()

    P1 = p1.shape[1]
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((p1.shape[0],), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((p1.shape[0],), P2, dtype=torch.int64, device=p1.device)

    # pyre-fixme[16]: `_knn_points` has no attribute `apply`.
    p1_dists, p1_idx = _knn_points.apply(
        p1, p2, lengths1, lengths2, K, version, return_sorted
    )

    p2_nn = None
    if return_nn:
        p2_nn = knn_gather(p2, p1_idx, lengths2)

    return _KNN(dists=p1_dists, idx=p1_idx, knn=p2_nn if return_nn else None)


def knn_gather(
    x: torch.Tensor, idx: torch.Tensor, lengths: Union[torch.Tensor, None] = None
):
    """
    A helper function for knn that allows indexing a tensor x with the indices `idx`
    returned by `knn_points`.
    For example, if `dists, idx = knn_points(p, x, lengths_p, lengths, K)`
    where p is a tensor of shape (N, L, D) and x a tensor of shape (N, M, D),
    then one can compute the K nearest neighbors of p with `p_nn = knn_gather(x, idx, lengths)`.
    It can also be applied for any tensor x of shape (N, M, U) where U != D.
    Args:
        x: Tensor of shape (N, M, U) containing U-dimensional features to
            be gathered.
        idx: LongTensor of shape (N, L, K) giving the indices returned by `knn_points`.
        lengths: LongTensor of shape (N,) of values in the range [0, M], giving the
            length of each example in the batch in x. Or None to indicate that every
            example has length M.
    Returns:
        x_out: Tensor of shape (N, L, K, U) resulting from gathering the elements of x
            with idx, s.t. `x_out[n, l, k] = x[n, idx[n, l, k]]`.
            If `k > lengths[n]` then `x_out[n, l, k]` is filled with 0.0.
    """
    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full((x.shape[0],), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    # idx_expanded has shape [N, L, K, U]

    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)
    # p2_nn has shape [N, L, K, U]

    needs_mask = lengths.min() < K
    if needs_mask:
        # mask has shape [N, K], true where idx is irrelevant because
        # there is less number of points in p2 than K
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]

        # expand mask to shape [N, L, K, U]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out


if __name__ == "__main__":
    custom_loss = MVLoss([1,1,1])

    original_target = torch.tensor([[1,2,3], [1,2,-3], [1,0,3], [2,4,5]], dtype=torch.float32,requires_grad=True)
    original_output = torch.tensor([[1,2,3], [1,3,-7], [1,5,9], [1,4,5]], dtype=torch.float32, requires_grad=True)

    output = original_output.unsqueeze(0)
    # print(output.shape)
    print(f"Model output: {output}\n")

    target = original_target.unsqueeze(0)
    # print(target.shape)
    print(f"Target sample: {target}\n")
    custom_loss(output, target)