import pytest
import torch
from torch import nn

from shannonist.models import BilinearPotential, MLP, MultiLinear, MultiMLP


def test_mlp_shape_initialization_and_empty_hidden_layers() -> None:
    model = MLP(6, output_dim=3, hidden_dim=(8, 4))
    output = model(torch.randn(5, 2, 3))
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]

    assert output.shape == (5, 3)
    assert [tuple(layer.weight.shape) for layer in linears] == [
        (8, 6),
        (4, 8),
        (3, 4),
    ]
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in linears)
    assert MLP(6, output_dim=2, hidden_dim=())(torch.randn(4, 6)).shape == (4, 2)


def test_bilinear_potential_concatenates_inputs() -> None:
    potential = BilinearPotential(3, output_dim=2, hidden_dim=())
    linear = next(
        module for module in potential.mlp.modules() if isinstance(module, nn.Linear)
    )
    x = torch.randn(4, 3)
    y = torch.randn(4, 3)

    assert linear.in_features == 6
    assert torch.allclose(potential(x, y), linear(torch.cat((x, y), dim=1)))


def test_multilinear_matches_independent_linear_layers() -> None:
    layer = MultiLinear(in_features=3, out_features=2, count=4)
    x = torch.randn(2, 5, 4, 3)
    expected = torch.stack(
        [
            torch.nn.functional.linear(x[..., index, :], layer.weight[index], layer.bias[index])
            for index in range(layer.count)
        ],
        dim=-2,
    )

    assert layer.weight.shape == (4, 2, 3)
    assert layer.bias.shape == (4, 2)
    assert torch.allclose(layer(x), expected)


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(3),
        torch.randn(2, 3, 3),
        torch.randn(2, 4, 5),
    ],
)
def test_multilinear_rejects_invalid_input_shapes(x: torch.Tensor) -> None:
    layer = MultiLinear(3, 2, count=4)
    with pytest.raises(ValueError):
        layer(x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_features": 0, "out_features": 2, "count": 2},
        {"in_features": 2, "out_features": 0, "count": 2},
        {"in_features": 2, "out_features": 2, "count": 0},
    ],
)
def test_multilinear_validates_dimensions(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        MultiLinear(**kwargs)


def test_multimlp_matches_independent_mlps() -> None:
    count = 3
    multi = MultiMLP(4, count=count, output_dim=2, hidden_dim=(5,))
    singles = [MLP(4, output_dim=2, hidden_dim=(5,)) for _ in range(count)]
    multi_linears = [
        module for module in multi._main if isinstance(module, MultiLinear)
    ]

    for index, single in enumerate(singles):
        single_linears = [
            module for module in single._main if isinstance(module, nn.Linear)
        ]
        for multi_layer, single_layer in zip(multi_linears, single_linears):
            single_layer.weight.data.copy_(multi_layer.weight.data[index])
            single_layer.bias.data.copy_(multi_layer.bias.data[index])

    x = torch.randn(2, 6, count, 4)
    expected = torch.stack(
        [
            singles[index](x[..., index, :].reshape(-1, 4)).reshape(2, 6, 2)
            for index in range(count)
        ],
        dim=-2,
    )
    output = multi(x)

    assert output.shape == (2, 6, count, 2)
    assert torch.allclose(output, expected, atol=1e-6)
    output.sum().backward()
    assert multi_linears[0].weight.grad is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0, "count": 2},
        {"input_dim": 2, "count": 0},
        {"input_dim": 2, "count": 2, "output_dim": 0},
        {"input_dim": 2, "count": 2, "hidden_dim": (3, 0)},
    ],
)
def test_multimlp_validates_dimensions(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MultiMLP(**kwargs)
