from collections.abc import Sequence
from copy import deepcopy

from torch import Tensor
from torch import nn


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
