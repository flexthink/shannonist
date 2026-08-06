import pytest
import torch
from torch import nn

from shannonist.models import (
    BilinearCritic,
    BilinearCriticOutput,
    MLP,
    MultiMLP,
    PairwiseCritic,
    SymmetricPairwiseCritic,
)


class DotPotential(nn.Module):
    """Return aligned dot products as column vectors."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x * y).sum(dim=-1, keepdim=True)


def test_bilinear_critic_normalization_temperature_and_output() -> None:
    x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    y = torch.tensor([[0.0, 2.0], [8.0, 15.0]])
    critic = BilinearCritic(
        nn.Identity(),
        nn.Identity(),
        DotPotential(),
        tau=4.0,
    )

    output = critic(x, y)

    assert isinstance(output, BilinearCriticOutput)
    assert output.batch_size == torch.Size([2])
    assert torch.allclose(output.hx.norm(dim=1), torch.full((2,), 0.5))
    assert torch.allclose(output.hy.norm(dim=1), torch.full((2,), 0.5))
    assert output.u.shape == (2, 1)


def test_bilinear_critic_can_skip_normalization() -> None:
    x = torch.randn(3, 2)
    y = torch.randn(3, 2)
    critic = BilinearCritic(
        nn.Identity(),
        nn.Identity(),
        DotPotential(),
        tau=4.0,
        use_norm=False,
    )

    output = critic(x, y)

    assert torch.allclose(output.hx, x / 2)
    assert torch.allclose(output.hy, y / 2)
    assert torch.allclose(output.u, (x * y).sum(dim=1, keepdim=True))


def test_pairwise_critic_matches_explicit_interactions() -> None:
    count = 4
    encoder = MultiMLP(3, count=count, output_dim=5, hidden_dim=(6,))
    critic = PairwiseCritic(encoder, count=count, use_norm=False)
    x = torch.randn(2, 7, count, 3)
    hx = critic.encode(x)
    expected = torch.empty(2, 7, count, count)

    for i in range(count):
        for j in range(count):
            expected[..., i, j] = torch.einsum(
                "...f,fg,...g->...",
                hx[..., i, :],
                critic.weight,
                hx[..., j, :],
            )

    output = critic.compute_interactions(hx)
    assert output.shape == (2, 7, count, count)
    assert torch.allclose(output, expected, atol=1e-5)
    assert torch.equal(output, output.transpose(-2, -1))
    assert torch.allclose(critic.weight, critic.weight.T)
    assert torch.linalg.eigvalsh(critic.weight).min() >= -1e-6

    output.sum().backward()
    assert critic.A.grad is not None
    assert next(encoder.parameters()).grad is not None


def test_pairwise_critic_normalizes_last_dimension() -> None:
    count = 3
    critic = PairwiseCritic(
        MultiMLP(2, count=count, output_dim=4, hidden_dim=()),
        count=count,
    )
    hx = critic.encode(torch.randn(5, count, 2))

    assert torch.allclose(hx.norm(dim=-1), torch.ones(5, count), atol=1e-6)


def test_pairwise_critic_validates_count_and_shapes() -> None:
    encoder = MultiMLP(2, count=3, output_dim=4)
    with pytest.raises(ValueError, match="does not match"):
        PairwiseCritic(encoder, count=2)

    critic = PairwiseCritic(encoder, count=3)
    with pytest.raises(ValueError, match="expected count"):
        critic(torch.randn(5, 2, 2))
    with pytest.raises(ValueError, match="encoded input"):
        critic.compute_interactions(torch.randn(5, 3, 5))


@pytest.mark.parametrize("count", [1, 3, 7])
def test_symmetric_pairwise_critic_supports_arbitrary_counts(count: int) -> None:
    encoder = MLP(3, output_dim=5, hidden_dim=(6,))
    critic = SymmetricPairwiseCritic(encoder, use_norm=False)
    x = torch.randn(2, 4, count, 3)

    hx = critic.encode(x)
    expected_hx = encoder(x.reshape(-1, 3)).reshape(2, 4, count, 5)
    output = critic.compute_interactions(hx)

    assert torch.allclose(hx, expected_hx)
    assert output.shape == (2, 4, count, count)
    assert torch.equal(output, output.transpose(-2, -1))
    assert torch.allclose(critic.weight, critic.weight.T)
    assert torch.linalg.eigvalsh(critic.weight).min() >= -1e-6

    output.sum().backward()
    assert critic.A.grad is not None
    assert next(encoder.parameters()).grad is not None


def test_symmetric_pairwise_critic_normalizes_and_validates_shapes() -> None:
    critic = SymmetricPairwiseCritic(MLP(2, output_dim=4, hidden_dim=()))
    hx = critic.encode(torch.randn(5, 6, 2))

    assert torch.allclose(hx.norm(dim=-1), torch.ones(5, 6), atol=1e-6)
    with pytest.raises(ValueError, match="feature dimension"):
        critic(torch.randn(5, 6, 3))
    with pytest.raises(ValueError, match="encoded input"):
        critic.compute_interactions(torch.randn(5, 6, 5))
