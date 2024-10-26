import math
import torch
from torch import nn

from .clifford_utils import unsqueeze_like


class MVLinear(nn.Module):
    # def __init__(
    #     self,
    #     algebra,
    #     in_features,
    #     out_features,
    #     subspaces=True,
    #     bias=True,
    # ):
    #     super().__init__()

    #     self.algebra = algebra
    #     self.in_features = in_features
    #     self.out_features = out_features
    #     self.subspaces = subspaces

    #     if subspaces:
    #         self.weight = nn.Parameter(
    #             torch.empty(out_features, in_features, algebra.n_subspaces)
    #         )
    #         self._forward = self._forward_subspaces
    #     else:
    #         self.weight = nn.Parameter(torch.empty(out_features, in_features))

    #     if bias:
    #         self.bias = nn.Parameter(torch.empty(1, out_features, 1))
    #         self.b_dims = (0,)
    #     else:
    #         self.register_parameter("bias", None)
    #         self.b_dims = ()

    #     self.reset_parameters()

    def __init__(self, algebra, in_features, out_features, subspaces=True, bias=True):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces
        self.subspace_dims = algebra.subspaces.tolist()  # assuming you have this attribute

        if subspaces:
            # Initialize weights for each subspace separately
            self.weight = nn.ParameterList([
                nn.Parameter(torch.empty(out_features, in_features, subspace_dim))
                for subspace_dim in self.subspace_dims
            ])
            self._forward = self._forward_subspaces
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features, 1))
            self.b_dims = (0,)
        else:
            self.register_parameter('bias', None)
            self.b_dims = ()

        self.reset_parameters()


    # def reset_parameters(self):
    #     torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

    #     if self.bias is not None:
    #         torch.nn.init.zeros_(self.bias)

    def reset_parameters(self):
        for weight in self.weight:
            torch.nn.init.normal_(weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def _forward(self, input):
        return torch.einsum("bm...i, nm->bn...i", input, self.weight)

    # def _forward_subspaces(self, input):
    #     weight = self.weight.repeat_interleave(self.algebra.subspaces, dim=-1)
    #     # print("--------")
    #     # print(input.shape)
    #     # print(weight.shape)
    #     return torch.einsum("bm...i, nmi->bn...i", input, weight)

    def _forward_subspaces(self, input):
        results = []
        start_dim = 0
        for weight, subspace_dim in zip(self.weight, self.subspace_dims):
            relevant_input = input[:, :, start_dim:start_dim+subspace_dim]
            # print("--------")
            # print(relevant_input.shape)
            # print(weight.shape)
            result = torch.einsum("bmi, nmi -> bni", relevant_input, weight)
            results.append(result)
            start_dim += subspace_dim
        full_result = torch.cat(results, dim=-1)
        return full_result


    def forward(self, input):
        result = self._forward(input)

        if self.bias is not None:
            bias = self.algebra.embed(self.bias, self.b_dims)
            result += unsqueeze_like(bias, result, dim=2)
        return result