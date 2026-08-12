import pytest
import torch
from tensordict import TensorDict
from torch import nn

from shannonist.mi import (
    JointFLO,
    ContrastivePairwiseFLO,
    ContrastivePairwiseFLOOutput,
    MIBatch,
    PairwiseFLO,
    PairwiseFLOOutput,
    PairwiseMIBatch,
    contrastive_pairwise_flo_loss,
    flo_loss,
    pairwise_flo_loss,
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


def make_joint_flo(
    input_dim: int = 3,
    feature_dim: int = 4,
) -> JointFLO:
    """Construct a small bilinear estimator for tests."""
    return JointFLO(
        MLP(input_dim, output_dim=feature_dim, hidden_dim=(6,)),
        MLP(input_dim, output_dim=feature_dim, hidden_dim=(6,)),
        BilinearPotential(feature_dim, hidden_dim=(5,)),
    )


def test_joint_flo_forward_and_objective_are_separate() -> None:
    encoder_x = CountingIdentity()
    encoder_y = CountingIdentity()
    model = JointFLO(
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
    assert torch.allclose(objective.estimate, -objective.loss)
    assert objective.metrics["similarity"].shape == (5, 5)

    objective.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_flo_loss_matches_explicit_formula() -> None:
    model = make_joint_flo()
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


def test_joint_flo_estimate_is_negative_objective() -> None:
    model = make_joint_flo()
    batch = MIBatch(
        x=torch.randn(7, 3),
        y=torch.randn(7, 3),
        batch_size=[7],
    )
    predictions = model.compute_forward(batch)
    objective = model.compute_objectives(predictions)
    estimate = model.estimate(batch)

    assert torch.allclose(estimate.value, -objective.loss)
    assert torch.allclose(estimate.value, objective.estimate)
    assert estimate.details["similarity"].shape == (7, 7)


def test_joint_flo_rejects_masks_and_single_sample() -> None:
    model = make_joint_flo()
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


def pairwise_flo_loss_ref(
    predictions: PairwiseFLOOutput,
) -> tuple[torch.Tensor, TensorDict]:
    """Reference loop implementation of masked pairwise FLO."""
    hx = predictions.hx
    u = predictions.u
    mask = predictions.mask
    if hx.ndim < 3:
        raise ValueError("hx must have shape (*, count, feature_dim)")
    if u.shape != (*hx.shape[:-1], hx.shape[-2]):
        raise ValueError("u must have shape (*, count, count)")
    if mask.shape != hx.shape[:-1]:
        raise ValueError("mask must have shape (*, count)")

    count = hx.shape[-2]
    feature_dim = hx.shape[-1]
    hx = hx.reshape(-1, count, feature_dim)
    u = u.reshape(-1, count, count)
    mask = mask.reshape(-1, count).bool()
    sample_count = hx.shape[0]
    zero = hx.sum() * 0
    pair_losses: dict[tuple[int, int], torch.Tensor] = {}
    loss_vec = hx.new_zeros(sample_count, count, count)
    valid_counts = torch.zeros(count, count, dtype=torch.long, device=hx.device)

    for i in range(count):
        for j in range(i + 1, count):
            valid = mask[:, i] & mask[:, j]
            valid_count = int(valid.sum().item())
            if valid_count < 2:
                raise ValueError(
                    f"pair ({i}, {j}) has fewer than two jointly valid samples"
                )

            hx_i = hx[valid, i]
            hx_j = hx[valid, j]
            potential = u[valid, i, j]
            directional_vectors = []
            for left, right in ((hx_i, hx_j), (hx_j, hx_i)):
                pair_similarity = left @ right.transpose(0, 1)
                positive_mask = torch.eye(
                    valid_count,
                    dtype=torch.bool,
                    device=hx.device,
                )
                g = pair_similarity[positive_mask]
                g0 = pair_similarity[~positive_mask].reshape(
                    valid_count,
                    valid_count - 1,
                )
                directional_vectors.append(
                    potential
                    + torch.exp(
                        -potential + torch.logsumexp(g0, dim=1) - g
                    )
                    / (valid_count - 1)
                    - 1
                )

            pair_loss_vec = torch.stack(directional_vectors).mean(dim=0)
            pair_loss = pair_loss_vec.mean()
            pair_losses[(i, j)] = pair_loss
            loss_vec[valid, i, j] = pair_loss_vec
            loss_vec[valid, j, i] = pair_loss_vec
            valid_counts[i, j] = valid_count
            valid_counts[j, i] = valid_count

    loss_matrix = torch.stack(
        [
            torch.stack(
                [
                    zero
                    if i == j
                    else pair_losses[(min(i, j), max(i, j))]
                    for j in range(count)
                ]
            )
            for i in range(count)
        ]
    )
    loss = torch.stack(list(pair_losses.values())).mean()
    similarity = torch.einsum("nif,mjf->ijnm", hx, hx)
    details = TensorDict(
        {
            "loss_vec": loss_vec,
            "loss_matrix": loss_matrix,
            "similarity": similarity,
            "u": u,
            "mask": mask,
            "valid_counts": valid_counts,
        },
        batch_size=[],
    )
    return loss, details


def test_vectorized_pairwise_flo_loss_matches_masked_reference() -> None:
    model = make_pairwise_flo(count=4)
    mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    batch = PairwiseMIBatch(
        x=torch.randn(6, 4, 4),
        x_mask=mask,
        batch_size=[6],
    )
    predictions = model.compute_forward(batch)

    actual_loss, actual_details = pairwise_flo_loss(predictions)
    reference_loss, reference_details = pairwise_flo_loss_ref(predictions)

    assert torch.allclose(actual_loss, reference_loss, atol=1e-6)
    assert set(actual_details.keys()) == set(reference_details.keys())
    for key in actual_details.keys():
        actual = actual_details[key]
        reference = reference_details[key]
        if actual.dtype == torch.bool or not actual.is_floating_point():
            assert torch.equal(actual, reference), key
        else:
            assert torch.allclose(actual, reference, atol=1e-6), key


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


def test_contrastive_pairwise_flo_samples_masked_sequences() -> None:
    torch.manual_seed(7)
    model = ContrastivePairwiseFLO(
        MLP(4, output_dim=5, hidden_dim=(6,)),
        sample_size=4,
    )
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    batch = PairwiseMIBatch(
        x=torch.randn(3, 6, 4),
        x_mask=mask.unsqueeze(-1),
        batch_size=[3],
    )

    predictions = model.compute_forward(batch)
    objective = model.compute_objectives(predictions)

    assert isinstance(predictions, ContrastivePairwiseFLOOutput)
    assert predictions.batch_size == torch.Size([3, 3])
    assert predictions.hx.shape == (3, 3, 2, 5)
    assert predictions.u.shape == (3, 3, 2, 2)
    assert predictions.position_indices.shape == (3, 3, 2)
    selected_mask = mask[
        torch.arange(3)[:, None, None],
        predictions.position_indices,
    ]
    assert selected_mask.all()
    assert torch.all(
        predictions.position_indices[..., 0]
        != predictions.position_indices[..., 1]
    )
    valid_ordinals = mask.cumsum(dim=-1) - 1
    sampled_ordinals = valid_ordinals[
        torch.arange(3)[:, None, None],
        predictions.position_indices,
    ]
    assert torch.equal(
        sampled_ordinals,
        sampled_ordinals[:1].expand_as(sampled_ordinals),
    )
    assert objective.loss.requires_grad
    assert objective.metrics["loss_vec"].shape == (3, 3)
    assert objective.metrics["similarity"].shape == (3, 2, 3, 3)

    objective.loss.backward()
    assert model.critic.A.grad is not None
    assert next(model.critic.encoder.parameters()).grad is not None


def test_contrastive_pairwise_flo_loss_matches_explicit_formula() -> None:
    model = ContrastivePairwiseFLO(
        MLP(3, output_dim=4, hidden_dim=()),
        sample_size=2,
        use_norm=False,
    )
    batch = PairwiseMIBatch(x=torch.randn(4, 5, 3), batch_size=[4])
    predictions = model.compute_forward(batch)
    loss, details = contrastive_pairwise_flo_loss(predictions)
    directional_losses = []
    independent = ~torch.eye(4, dtype=torch.bool)

    for pair_index in range(2):
        pair_directions = []
        potential = predictions.u[:, pair_index, 0, 1]
        for left, right in ((0, 1), (1, 0)):
            similarity = (
                predictions.hx[:, pair_index, left]
                @ predictions.hx[:, pair_index, right].T
            )
            joint = similarity.diag()
            negatives = similarity[independent].reshape(4, 3)
            pair_directions.append(
                potential
                + torch.exp(
                    -potential
                    + torch.logsumexp(negatives, dim=-1)
                    - joint
                )
                / 3
                - 1
            )
        directional_losses.append(torch.stack(pair_directions).mean(dim=0))

    expected = torch.stack(directional_losses, dim=1)
    assert torch.allclose(details["loss_vec"], expected, atol=1e-6)
    assert torch.allclose(loss, expected.mean(), atol=1e-6)


def test_contrastive_pairwise_flo_validates_batch_masks_and_sample_size() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        ContrastivePairwiseFLO(MLP(2, output_dim=3), sample_size=0)

    model = ContrastivePairwiseFLO(MLP(2, output_dim=3), sample_size=2)
    single = PairwiseMIBatch(x=torch.randn(1, 4, 2), batch_size=[1])
    with pytest.raises(ValueError, match="batch size"):
        model.compute_forward(single)

    empty = PairwiseMIBatch(
        x=torch.randn(2, 4, 2),
        x_mask=torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]),
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="at least one valid"):
        model.compute_forward(empty)

    invalid_mask = PairwiseMIBatch(
        x=torch.randn(2, 4, 2),
        x_mask=torch.ones(2, 3, dtype=torch.bool),
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="x_mask"):
        model.compute_forward(invalid_mask)
