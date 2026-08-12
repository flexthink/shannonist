import pytest
import torch

from shannonist.models import (
    FlowDensityEstimator,
    Invertible,
    InvertibleLinear,
    InvertibleMLP,
    InvertibleOutput,
)


def test_invertible_requires_an_inverse() -> None:
    class ForwardOnly(Invertible):
        def forward(self, x: torch.Tensor) -> InvertibleOutput:
            return InvertibleOutput(
                value=x,
                log_abs_det=x.new_zeros(x.shape[:-1]),
                batch_size=x.shape[:-1],
            )

    with pytest.raises(TypeError):
        ForwardOnly()


def test_invertible_subclass_can_round_trip() -> None:
    class Shift(Invertible):
        def forward(self, x: torch.Tensor) -> InvertibleOutput:
            return InvertibleOutput(
                value=x + 1,
                log_abs_det=x.new_zeros(x.shape[:-1]),
                batch_size=x.shape[:-1],
            )

        def inverse(self, y: torch.Tensor) -> InvertibleOutput:
            return InvertibleOutput(
                value=y - 1,
                log_abs_det=y.new_zeros(y.shape[:-1]),
                batch_size=y.shape[:-1],
            )

    transform = Shift()
    x = torch.randn(4, 3)

    transformed = transform(x)
    reconstructed = transform.inverse(transformed.value)
    assert torch.allclose(reconstructed.value, x)


def test_invertible_linear_matches_linear_semantics() -> None:
    transform = InvertibleLinear(dim=3)
    x = torch.randn(5, 3)

    expected = torch.nn.functional.linear(
        x,
        transform.weight,
        transform.bias,
    )

    output = transform(x)
    assert torch.allclose(output.value, expected)


@pytest.mark.parametrize("bias", [True, False])
def test_invertible_linear_round_trip(bias: bool) -> None:
    transform = InvertibleLinear(dim=4, bias=bias)
    x = torch.randn(2, 3, 4)

    transformed = transform(x)
    reconstructed = transform.inverse(transformed.value)

    assert reconstructed.value.shape == x.shape
    assert torch.allclose(reconstructed.value, x, atol=1e-6)
    assert torch.allclose(
        transformed.log_abs_det + reconstructed.log_abs_det,
        torch.zeros_like(transformed.log_abs_det),
    )


def test_invertible_linear_inverse_is_differentiable() -> None:
    transform = InvertibleLinear(dim=3)
    y = torch.randn(4, 3, requires_grad=True)

    transform.inverse(y).value.sum().backward()

    assert y.grad is not None
    assert transform.l_params.grad is not None
    assert transform.u_params.grad is not None
    assert transform.log_diag.grad is not None
    assert transform.bias.grad is not None


def test_invertible_linear_initialization() -> None:
    transform = InvertibleLinear(dim=4)
    identity = torch.eye(4)

    assert torch.allclose(
        transform.weight @ transform.weight.T,
        identity,
        atol=1e-6,
    )
    assert torch.allclose(transform.P @ transform.P.T, identity)
    assert torch.all(transform.U().diagonal() != 0)
    assert torch.equal(transform.bias, torch.zeros(4))


def test_invertible_linear_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="dim"):
        InvertibleLinear(dim=0)

    transform = InvertibleLinear(dim=3)
    with pytest.raises(ValueError, match="trailing dimension"):
        transform(torch.randn(4, 2))
    with pytest.raises(ValueError, match="trailing dimension"):
        transform.inverse(torch.randn(4, 2))


def test_invertible_linear_remains_invertible_after_parameter_updates() -> None:
    transform = InvertibleLinear(dim=3)
    with torch.no_grad():
        transform.l_params.normal_()
        transform.u_params.normal_()
        transform.log_diag.copy_(torch.tensor([-10.0, 0.0, 10.0]))
    x = torch.randn(5, 3)

    assert torch.all(transform.U().diagonal() != 0)
    assert torch.allclose(
        transform.inverse(transform(x).value).value,
        x,
        atol=2e-2,
        rtol=2e-2,
    )


def test_invertible_linear_ignores_unused_triangle_parameters() -> None:
    transform = InvertibleLinear(dim=3)
    with torch.no_grad():
        transform.P.copy_(torch.eye(3))
        transform.sign_diag.fill_(1)
        transform.log_diag.zero_()
        transform.l_params.copy_(torch.triu(torch.ones(3, 3)))
        transform.u_params.copy_(torch.tril(torch.ones(3, 3)))

    assert torch.equal(transform.L(), torch.eye(3))
    assert torch.equal(transform.U(), torch.eye(3))


def test_invertible_mlp_round_trip() -> None:
    transform = InvertibleMLP(
        input_dim=4,
        hidden_dims=(4, 4),
        output_dim=4,
    )
    x = torch.randn(2, 3, 4)

    output = transform(x)
    reconstructed = transform.inverse(output.value)

    assert output.value.shape == x.shape
    assert len(transform.layers) == 3
    assert torch.allclose(reconstructed.value, x, atol=1e-5)
    assert torch.allclose(
        output.log_abs_det + reconstructed.log_abs_det,
        torch.zeros_like(output.log_abs_det),
    )


def test_invertible_mlp_applies_inverse_layers_in_reverse() -> None:
    transform = InvertibleMLP(
        input_dim=3,
        hidden_dims=(3,),
        output_dim=3,
    )
    x = torch.randn(5, 3)
    expected = x
    expected_log_abs_det = x.new_zeros(x.shape[:-1])
    for layer in reversed(transform.layers):
        output = layer.inverse(expected)
        expected = output.value
        expected_log_abs_det += output.log_abs_det

    output = transform.inverse(x)
    assert torch.allclose(output.value, expected)
    assert torch.allclose(output.log_abs_det, expected_log_abs_det)


def test_invertible_linear_reports_log_absolute_determinant() -> None:
    transform = InvertibleLinear(dim=3)
    with torch.no_grad():
        transform.log_diag.copy_(torch.tensor([-1.0, 0.5, 2.0]))
    x = torch.randn(2, 4, 3)

    forward = transform(x)
    inverse = transform.inverse(forward.value)

    expected = torch.full((2, 4), 1.5)
    assert forward.batch_size == torch.Size([2, 4])
    assert torch.allclose(forward.log_abs_det, expected)
    assert torch.allclose(inverse.log_abs_det, -expected)


def test_invertible_mlp_sums_layer_log_determinants() -> None:
    transform = InvertibleMLP(3, (3, 3), 3)
    with torch.no_grad():
        for index, layer in enumerate(transform.layers, start=1):
            layer.log_diag.fill_(index / 3)
    x = torch.randn(5, 3)

    output = transform(x)

    expected = sum(layer.log_abs_det for layer in transform.layers).expand(5)
    assert torch.allclose(output.log_abs_det, expected)


def test_invertible_mlp_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        InvertibleMLP(input_dim=0, hidden_dims=(), output_dim=0)
    with pytest.raises(ValueError, match="must match"):
        InvertibleMLP(input_dim=3, hidden_dims=(4,), output_dim=3)
    with pytest.raises(ValueError, match="must match"):
        InvertibleMLP(input_dim=3, hidden_dims=(3,), output_dim=2)


def make_affine_flow_density() -> FlowDensityEstimator:
    """Construct a two-dimensional flow with a known diagonal weight."""
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)),
        1,
    )
    transform = InvertibleLinear(dim=2)
    with torch.no_grad():
        transform.P.copy_(torch.eye(2))
        transform.sign_diag.fill_(1)
        transform.l_params.zero_()
        transform.u_params.zero_()
        transform.log_diag.copy_(torch.log(torch.tensor([2.0, 3.0])))
        transform.bias.copy_(torch.tensor([1.0, -1.0]))
    return FlowDensityEstimator(transform=transform, prior=prior)


def test_flow_density_estimator_samples_and_transforms() -> None:
    estimator = make_affine_flow_density()

    output = estimator(sample_shape=(7,))

    assert output.batch_size == torch.Size([7])
    assert output.value.shape == output.latent.shape == (7, 2)
    assert torch.allclose(
        output.value,
        estimator.transform(output.latent).value,
    )
    expected_log_prob = estimator.prior.log_prob(output.latent) - torch.log(
        torch.tensor(6.0)
    )
    assert torch.allclose(output.log_prob, expected_log_prob)


def test_flow_density_estimator_evaluates_by_inversion() -> None:
    estimator = make_affine_flow_density()
    value = torch.tensor([[1.0, -1.0], [3.0, 2.0]])

    output = estimator.evaluate(value)

    expected_latent = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    expected_log_prob = estimator.prior.log_prob(expected_latent) - torch.log(
        torch.tensor(6.0)
    )
    assert torch.allclose(output.latent, expected_latent)
    assert torch.allclose(output.log_prob, expected_log_prob)
    assert torch.allclose(estimator.log_prob(value), expected_log_prob)
    assert torch.allclose(estimator.prob(value), expected_log_prob.exp())


def test_flow_density_estimator_sample_and_evaluation_agree() -> None:
    estimator = make_affine_flow_density()

    sampled = estimator(sample_shape=(16,))
    evaluated = estimator.evaluate(sampled.value)

    assert torch.allclose(evaluated.latent, sampled.latent, atol=1e-6)
    assert torch.allclose(evaluated.log_prob, sampled.log_prob, atol=1e-6)
    assert torch.allclose(evaluated.log_abs_det, -sampled.log_abs_det)


def test_flow_density_estimator_log_prob_is_differentiable() -> None:
    estimator = make_affine_flow_density()
    value = torch.randn(5, 2)

    loss = -estimator.log_prob(value).mean()
    loss.backward()

    assert estimator.transform.log_diag.grad is not None
    assert estimator.transform.bias.grad is not None


def test_flow_density_estimator_validates_components_and_shapes() -> None:
    prior = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    transform = InvertibleLinear(dim=2)

    with pytest.raises(TypeError, match="Distribution"):
        FlowDensityEstimator(transform=transform, prior=object())
    with pytest.raises(TypeError, match="Invertible"):
        FlowDensityEstimator(transform=torch.nn.Identity(), prior=prior)

    estimator = FlowDensityEstimator(transform=transform, prior=prior)
    with pytest.raises(ValueError, match="identical shapes"):
        estimator.evaluate(torch.randn(4, 2))


def test_flow_density_estimator_defaults_to_standard_gaussian() -> None:
    transform = InvertibleLinear(dim=3)
    estimator = FlowDensityEstimator(transform=transform)
    value = torch.zeros(4, 3)

    output = estimator.evaluate(value)

    expected = torch.full(
        (4,),
        -1.5 * torch.log(torch.tensor(2 * torch.pi)),
    )
    assert torch.allclose(output.log_prob, expected, atol=1e-6)
    assert estimator.prior.event_shape == torch.Size([3])


def test_default_flow_prior_follows_module_dtype() -> None:
    estimator = FlowDensityEstimator(InvertibleLinear(dim=2)).double()

    output = estimator(sample_shape=(3,))

    assert output.value.dtype == torch.float64
    assert output.latent.dtype == torch.float64


def test_default_flow_prior_requires_transform_dimension() -> None:
    class DimensionlessInvertible(Invertible):
        def forward(self, x: torch.Tensor) -> InvertibleOutput:
            raise NotImplementedError

        def inverse(self, y: torch.Tensor) -> InvertibleOutput:
            raise NotImplementedError

    with pytest.raises(ValueError, match="expose.*dim"):
        FlowDensityEstimator(DimensionlessInvertible())
