from collections.abc import Sequence
from copy import deepcopy

import torch
from torch import Tensor
from torch import nn


class MultiLinear(nn.Module):
    """Apply independent linear transformations along a count dimension.

    Parameters are stored as a single weight tensor, but each position in the
    count dimension has its own linear transformation.

    Parameters
    ----------
    in_features : int
        Number of input features for each transformation.
    out_features : int
        Number of output features for each transformation.
    count : int
        Number of independent linear transformations.
    bias : bool, default=True
        Whether to learn an independent bias for each transformation.

    Attributes
    ----------
    weight : nn.Parameter
        Weight tensor with shape ``(count, out_features, in_features)``.
    bias : nn.Parameter or None
        Bias tensor with shape ``(count, out_features)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        count: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if count <= 0:
            raise ValueError("count must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.count = count
        self.weight = nn.Parameter(torch.empty(count, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(count, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every transformation like an MLP linear layer."""
        for weight in self.weight:
            nn.init.xavier_uniform_(weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the independent linear transformations.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, count, in_features)``.

        Returns
        -------
        Tensor
            Output with shape ``(*, count, out_features)``.

        Raises
        ------
        ValueError
            If the count or feature dimension does not match the layer.
        """
        if x.ndim < 2:
            raise ValueError("input must have shape (*, count, in_features)")
        if x.shape[-2] != self.count:
            raise ValueError(
                f"expected count dimension {self.count}, got {x.shape[-2]}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected feature dimension {self.in_features}, got {x.shape[-1]}"
            )

        output = torch.einsum("...ci,coi->...co", x, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class MLP(nn.Module):
    """Multilayer perceptron with configurable hidden dimensions.

    Each hidden linear layer is followed by an activation function. All linear
    weights use Xavier-uniform initialization and all biases are initialized to
    zero. Inputs are flattened after the batch dimension before being passed
    through the network.

    Parameters
    ----------
    input_dim : int
        Number of input features after flattening.
    output_dim : int, default=1
        Number of output features.
    hidden_dim : Sequence[int], default=(512, 512)
        Width of each hidden layer. An empty sequence creates a single linear
        layer from ``input_dim`` to ``output_dim``.
    act_func : nn.Module, optional
        Activation applied after each hidden layer. Defaults to ``nn.ReLU``.

    References
    ----------
    Adapted from the FLO bilinear critic code accompanying Guo et al.,
    "Tight Mutual Information Estimation With Contrastive Fenchel-Legendre
    Optimization," NeurIPS 2022: https://arxiv.org/abs/2107.01131
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: Sequence[int] = (512, 512),
        act_func: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = tuple(hidden_dim)
        self.output_dim = output_dim
        self.act_func = act_func if act_func is not None else nn.ReLU()

        dimensions = (input_dim, *self.hidden_dim, output_dim)
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            layer = nn.Linear(in_features, out_features)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)

            if index < len(self.hidden_dim):
                activation = self.act_func if index == 0 else deepcopy(self.act_func)
                layers.append(activation)

        self._main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the multilayer perceptron.

        Parameters
        ----------
        x : Tensor
            Input whose first dimension is the batch dimension. All remaining
            dimensions must contain ``input_dim`` elements in total.

        Returns
        -------
        Tensor
            Network output with shape ``(batch_size, output_dim)``.
        """
        flattened = x.reshape(x.shape[0], self.input_dim)
        return self._main(flattened)


class MultiMLP(nn.Module):
    """Evaluate multiple independent MLPs in parallel.

    ``MultiMLP`` is mathematically equivalent to evaluating ``count`` MLPs,
    one for each position in the input's count dimension, and stacking their
    outputs along that same dimension. Its linear parameters are stored in
    :class:`MultiLinear` layers for vectorized evaluation.

    Parameters
    ----------
    input_dim : int
        Number of features supplied to each MLP.
    count : int
        Number of independent MLPs.
    output_dim : int, default=1
        Number of output features produced by each MLP.
    hidden_dim : Sequence[int], default=(512, 512)
        Width of each hidden layer. An empty sequence creates a single
        ``MultiLinear`` layer from ``input_dim`` to ``output_dim``.
    act_func : nn.Module, optional
        Activation applied after each hidden layer. Defaults to ``nn.ReLU``.
    """

    def __init__(
        self,
        input_dim: int,
        count: int,
        output_dim: int = 1,
        hidden_dim: Sequence[int] = (512, 512),
        act_func: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if count <= 0:
            raise ValueError("count must be positive")
        if any(dimension <= 0 for dimension in hidden_dim):
            raise ValueError("hidden dimensions must be positive")

        self.input_dim = input_dim
        self.count = count
        self.hidden_dim = tuple(hidden_dim)
        self.output_dim = output_dim
        self.act_func = act_func if act_func is not None else nn.ReLU()

        dimensions = (input_dim, *self.hidden_dim, output_dim)
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            layers.append(
                MultiLinear(
                    in_features=in_features,
                    out_features=out_features,
                    count=count,
                )
            )
            if index < len(self.hidden_dim):
                activation = self.act_func if index == 0 else deepcopy(self.act_func)
                layers.append(activation)

        self._main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate all independent MLPs.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(*, count, input_dim)``.

        Returns
        -------
        Tensor
            Output with shape ``(*, count, output_dim)``.
        """
        return self._main(x)


__all__ = ["MLP", "MultiLinear", "MultiMLP"]
