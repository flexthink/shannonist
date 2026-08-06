import pytest
import torch
from torch import nn

from shannonist.mi import (
    BilinearFLO,
    MIBatch,
    PairwiseFLO,
    PairwiseFLOOutput,
    PairwiseMIBatch,
    flo_loss,
)
from shannonist.models import BilinearPotential, MLP, MultiMLP


class CountingIdentity(nn.Module):
    """Identity encoder recording the number of forward calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x


def make_bilinear_flo(
    input_dim: int = 3,
    feature_dim: int = 4,
) -> BilinearFLO:
    """Construct a small bilinear estimator for tests."""
    return BilinearFLO(
        MLP(input_dim, output_dim=feature_dim, hidden_dim=(6,)),
        MLP(input_dim, output_dim=feature_dim, hidden_dim=(6,)),
        BilinearPotential(feature_dim, hidden_dim=(5,)),
    )


def test_bilinear_flo_forward_and_objective_are_separate() -> None:
    encoder_x = CountingIdentity()
    encoder_y = CountingIdentity()
    model = BilinearFLO(
        encoder_x,
        encoder_y,
        BilinearPotential(3, hidden_dim=(4,)),
    )
    batch = MIBatch(
        x=torch.randn(5, 3),
        y=torch.randn(5, 3),
        batch_size=[5],
    )

    predictions = model.compute_forward(batch)
    assert encoder_x.calls == encoder_y.calls == 1

    objective = model.compute_objectives(predictions)
    assert encoder_x.calls == encoder_y.calls == 1
    assert objective.loss.requires_grad
    assert objective.metrics["similarity"].shape == (5, 5)

    objective.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_flo_loss_matches_explicit_formula() -> None:
    model = make_bilinear_flo()
    batch = MIBatch(
        x=torch.randn(6, 3),
        y=torch.randn(6, 3),
        batch_size=[6],
    )
    predictions = model.compute_forward(batch)
    loss, details = flo_loss(predictions)
    hx = predictions.critic.hx
    hy = predictions.critic.hy
    u = predictions.critic.u
    similarity = hx @ hy.T
    mask = torch.eye(6, dtype=torch.bool)
    positives = similarity[mask].reshape(6, 1)
    negatives = similarity[~mask].reshape(6, 5)
    expected_vec = (
        u
        + torch.exp(-u + torch.logsumexp(negatives, 1, keepdim=True) - positives)
        / 5
        - 1
    )

    assert torch.allclose(loss, expected_vec.mean())
    assert torch.allclose(details["loss_vec"], expected_vec)


def test_bilinear_flo_estimate_is_negative_objective() -> None:
    model = make_bilinear_flo()
    batch = MIBatch(
        x=torch.randn(7, 3),
        y=torch.randn(7, 3),
        batch_size=[7],
    )
    predictions = model.compute_forward(batch)
    objective = model.compute_objectives(predictions)
    estimate = model.estimate(batch)

    assert torch.allclose(estimate.value, -objective.loss)
    assert estimate.details["similarity"].shape == (7, 7)


def test_bilinear_flo_rejects_masks_and_single_sample() -> None:
    model = make_bilinear_flo()
    masked = MIBatch(
        x=torch.randn(4, 3),
        y=torch.randn(4, 3),
        x_mask=torch.ones(4, 1, dtype=torch.bool),
        batch_size=[4],
    )
    with pytest.raises(NotImplementedError):
        model.compute_forward(masked)

    single = MIBatch(
        x=torch.randn(1, 3),
        y=torch.randn(1, 3),
        batch_size=[1],
    )
    with pytest.raises(ValueError, match="at least two"):
        model.compute_objectives(model.compute_forward(single))


def make_pairwise_flo(count: int = 3) -> PairwiseFLO:
    """Construct a small pairwise estimator for tests."""
    return PairwiseFLO(
        MultiMLP(4, count=count, output_dim=5, hidden_dim=(7,)),
        count=count,
    )


def test_pairwise_flo_shapes_symmetry_diagonal_and_gradients() -> None:
    count = 3
    model = make_pairwise_flo(count)
    batch = PairwiseMIBatch(
        x=torch.randn(12, count, 4),
        batch_size=[12],
    )

    predictions = model.compute_forward(batch)
    objective = model.compute_objectives(predictions)
    estimate = model.estimate(batch)

    assert isinstance(predictions, PairwiseFLOOutput)
    assert predictions.hx.shape == (12, count, 5)
    assert predictions.u.shape == (12, count, count)
    assert torch.all(predictions.mask)
    assert objective.loss.requires_grad
    assert estimate.value.shape == (count, count)
    assert torch.equal(estimate.value, estimate.value.T)
    assert torch.count_nonzero(estimate.value.diag()) == 0

    objective.loss.backward()
    assert model.critic.A.grad is not None
    assert next(model.critic.encoder.parameters()).grad is not None


def test_pairwise_flo_matches_mean_of_explicit_pair_losses() -> None:
    count = 3
    model = make_pairwise_flo(count)
    batch = PairwiseMIBatch(
        x=torch.randn(9, count, 4),
        batch_size=[9],
    )
    predictions = model.compute_forward(batch)
    objective = model.compute_objectives(predictions)
    hx = predictions.hx
    u = predictions.u
    expected_pair_losses = []
    negative_mask = ~torch.eye(len(hx), dtype=torch.bool)

    for i in range(count):
        for j in range(i + 1, count):
            directional_losses = []
            for left, right in ((i, j), (j, i)):
                similarity = hx[:, left] @ hx[:, right].T
                positives = similarity.diag().reshape(-1, 1)
                negatives = similarity[negative_mask].reshape(len(hx), -1)
                potential = u[:, left, right].reshape(-1, 1)
                loss = (
                    potential
                    + torch.exp(
                        -potential
                        + torch.logsumexp(negatives, 1, keepdim=True)
                        - positives
                    )
                    / (len(hx) - 1)
                    - 1
                ).mean()
                directional_losses.append(loss)
            expected_pair_losses.append(torch.stack(directional_losses).mean())

    assert torch.allclose(
        objective.loss,
        torch.stack(expected_pair_losses).mean(),
        atol=1e-6,
    )


def test_pairwise_flo_flattens_leading_sample_dimensions() -> None:
    model = make_pairwise_flo()
    batch = PairwiseMIBatch(
        x=torch.randn(2, 4, 3, 4),
        batch_size=[2, 4],
    )

    predictions = model.compute_forward(batch)
    estimate = model.estimate(batch)

    assert predictions.batch_size == torch.Size([2, 4])
    assert estimate.value.shape == (3, 3)


def test_pairwise_flo_validates_count_and_sample_count() -> None:
    with pytest.raises(ValueError, match="at least two"):
        make_pairwise_flo(count=1)

    model = make_pairwise_flo()
    batch = PairwiseMIBatch(x=torch.randn(1, 3, 4), batch_size=[1])
    with pytest.raises(ValueError, match="two jointly valid samples"):
        model.compute_objectives(model.compute_forward(batch))


def test_pairwise_flo_mask_is_equivalent_to_removing_samples() -> None:
    model = make_pairwise_flo(count=2)
    x = torch.randn(6, 2, 4)
    mask = torch.tensor(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 0],
        ],
        dtype=torch.bool,
    )
    masked_batch = PairwiseMIBatch(
        x=x,
        x_mask=mask.unsqueeze(-1),
        batch_size=[6],
    )
    valid = mask.all(dim=1)
    filtered_batch = PairwiseMIBatch(x=x[valid], batch_size=[int(valid.sum())])

    masked_predictions = model.compute_forward(masked_batch)
    masked_objective = model.compute_objectives(masked_predictions)
    filtered_objective = model.compute_objectives(model.compute_forward(filtered_batch))

    assert masked_predictions.mask.shape == (6, 2)
    assert torch.equal(masked_predictions.mask, mask)
    assert torch.allclose(masked_objective.loss, filtered_objective.loss)
    assert torch.allclose(
        masked_objective.metrics["loss_matrix"],
        filtered_objective.metrics["loss_matrix"],
    )
    assert masked_objective.metrics["valid_counts"][0, 1] == valid.sum()


def test_pairwise_flo_masked_values_do_not_affect_loss() -> None:
    model = make_pairwise_flo(count=2)
    x = torch.randn(5, 2, 4)
    mask = torch.tensor(
        [[1, 1], [1, 0], [1, 1], [0, 1], [1, 1]],
        dtype=torch.bool,
    )
    modified_x = x.clone()
    modified_x[~mask] = 1_000_000

    original = PairwiseMIBatch(x=x, x_mask=mask, batch_size=[5])
    modified = PairwiseMIBatch(x=modified_x, x_mask=mask, batch_size=[5])
    original_loss = model.compute_objectives(model.compute_forward(original)).loss
    modified_loss = model.compute_objectives(model.compute_forward(modified)).loss

    assert torch.allclose(original_loss, modified_loss)


def test_pairwise_flo_mask_tracks_valid_counts_per_pair() -> None:
    model = make_pairwise_flo(count=3)
    mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=torch.bool,
    )
    batch = PairwiseMIBatch(
        x=torch.randn(4, 3, 4),
        x_mask=mask,
        batch_size=[4],
    )
    objective = model.compute_objectives(model.compute_forward(batch))

    expected_counts = torch.tensor(
        [[0, 2, 2], [2, 0, 2], [2, 2, 0]],
    )
    assert torch.equal(objective.metrics["valid_counts"].cpu(), expected_counts)


def test_pairwise_flo_validates_mask_shape_and_pair_coverage() -> None:
    model = make_pairwise_flo(count=3)
    invalid_shape = PairwiseMIBatch(
        x=torch.randn(4, 3, 4),
        x_mask=torch.ones(4, 2, dtype=torch.bool),
        batch_size=[4],
    )
    with pytest.raises(ValueError, match="x_mask"):
        model.compute_forward(invalid_shape)

    insufficient = PairwiseMIBatch(
        x=torch.randn(4, 3, 4),
        x_mask=torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
            dtype=torch.bool,
        ),
        batch_size=[4],
    )
    with pytest.raises(ValueError, match=r"pair \(0, 1\)"):
        model.compute_objectives(model.compute_forward(insufficient))
