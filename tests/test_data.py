import math

import pytest
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from shannonist.mi import (
    ConditionedPairwiseCorrelatedGaussian,
    CorrelatedGausian,
    LatentPairwiseCorrelatedGaussian,
    MixtureLatentPairwiseCorrelatedGaussian,
    PairwiseCorrelatedGaussian,
    tensordict_collate,
    tensordict_passthrough,
)


def test_mixture_latent_pairwise_correlated_gaussian_context_bags() -> None:
    matrices = [
        [[0.0, 0.1], [0.1, 0.0]],
        [[0.0, 0.3], [0.3, 0.0]],
    ]
    dataset = MixtureLatentPairwiseCorrelatedGaussian(
        count=5,
        batch_size=3,
        mutual_information=matrices,
        dim=4,
        context_count=6,
        num_batches=2,
    )

    batch = dataset[0]

    assert batch.batch_size == torch.Size([3])
    assert batch["x"].shape == (3, 2, 4)
    assert batch["context"].shape == (3, 6, 4)
    assert batch["context_mask"].shape == (3, 6)
    assert batch["context_mask"].any(dim=-1).all()
    assert batch["regime"].unique().numel() == 1
    assert dataset.conditional_mutual_information.shape == (2, 2, 2)
    assert not torch.equal(
        dataset.conditional_mutual_information[0],
        dataset.conditional_mutual_information[1],
    )


def test_conditioned_pairwise_correlated_gaussian_shapes() -> None:
    matrices = [
        [[0.0, 0.1], [0.1, 0.0]],
        [[0.0, 0.3], [0.3, 0.0]],
    ]
    dataset = ConditionedPairwiseCorrelatedGaussian(
        matrices,
        dim=3,
        cond_dim=4,
        context_separation=0.5,
        context_std=1.0,
        num_samples=10,
    )

    sample = dataset[0]

    assert sample["x"].shape == (2, 3)
    assert sample["cond"].shape == (4,)
    assert sample["regime"].shape == ()
    assert sample["regime"].item() in {0, 1}
    assert dataset.mutual_information.shape == (2, 2, 2)


def test_correlated_gaussian_properties_and_dataloader_collation() -> None:
    target_mi = 2.0
    dim = 4
    dataset = CorrelatedGausian(target_mi, dim=dim, num_samples=12)
    expected_rho = math.sqrt(1 - math.exp(-2 * target_mi / dim))

    assert len(dataset) == 12
    assert dataset.rho == pytest.approx(expected_rho)
    sample = dataset[0]
    assert isinstance(sample, TensorDict)
    assert sample.batch_size == torch.Size([])
    assert set(sample.keys()) == {"x", "y"}

    batch = next(
        iter(DataLoader(dataset, batch_size=5, collate_fn=tensordict_collate))
    )
    assert isinstance(batch, TensorDict)
    assert batch.batch_size == torch.Size([5])
    assert batch["x"].shape == (5, dim)
    assert batch["y"].shape == (5, dim)


def test_correlated_gaussian_empirical_correlation() -> None:
    dataset = CorrelatedGausian(1.0, dim=3, num_samples=4_000)
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=len(dataset),
                collate_fn=tensordict_collate,
            )
        )
    )
    empirical = torch.corrcoef(
        torch.stack((batch["x"].flatten(), batch["y"].flatten()))
    )[0, 1]

    assert empirical.item() == pytest.approx(dataset.rho, abs=0.03)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mutual_information": -1.0}, "mutual_information"),
        ({"mutual_information": 1.0, "dim": 0}, "dim"),
        ({"mutual_information": 1.0, "num_samples": -1}, "num_samples"),
    ],
)
def test_correlated_gaussian_validates_constructor(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CorrelatedGausian(**kwargs)


def test_correlated_gaussian_validates_indices() -> None:
    dataset = CorrelatedGausian(0.0, num_samples=2)

    assert dataset[-1]["x"].shape == (1,)
    with pytest.raises(IndexError):
        dataset[2]
    with pytest.raises(IndexError):
        dataset[-3]


def test_tensordict_collate_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="empty"):
        tensordict_collate([])


def test_tensordict_passthrough_preserves_dataset_batch() -> None:
    batch = TensorDict({"x": torch.randn(3, 2)}, batch_size=[3])

    assert tensordict_passthrough(batch) is batch


def test_pairwise_correlated_gaussian_factor_and_shapes() -> None:
    mutual_information = torch.tensor(
        [
            [0.0, 0.2, 0.2],
            [0.2, 0.0, 0.2],
            [0.2, 0.2, 0.0],
        ]
    )
    dataset = PairwiseCorrelatedGaussian(
        mutual_information,
        dim=2,
        num_samples=10,
    )

    assert len(dataset) == 10
    assert dataset.count == 3
    sample = dataset[0]
    assert isinstance(sample, TensorDict)
    assert sample.batch_size == torch.Size([])
    assert sample["x"].shape == (3, 2)
    assert torch.allclose(
        dataset.factor @ dataset.factor.T,
        dataset.correlation,
        atol=1e-6,
    )
    assert torch.equal(dataset.correlation.diag(), torch.ones(3))

    batch = next(
        iter(DataLoader(dataset, batch_size=4, collate_fn=tensordict_collate))
    )
    assert isinstance(batch, TensorDict)
    assert batch.batch_size == torch.Size([4])
    assert batch["x"].shape == (4, 3, 2)


def test_pairwise_correlated_gaussian_empirical_mi() -> None:
    target = 0.3
    dim = 2
    mutual_information = torch.full((3, 3), target)
    mutual_information.fill_diagonal_(0)
    dataset = PairwiseCorrelatedGaussian(
        mutual_information,
        dim=dim,
        num_samples=6_000,
    )
    x = next(
        iter(
            DataLoader(
                dataset,
                batch_size=len(dataset),
                collate_fn=tensordict_collate,
            )
        )
    )["x"]

    for i in range(3):
        for j in range(i + 1, 3):
            correlation = torch.corrcoef(
                torch.stack((x[:, i].flatten(), x[:, j].flatten()))
            )[0, 1]
            empirical_mi = -dim / 2 * torch.log1p(-(correlation**2))
            assert empirical_mi.item() == pytest.approx(target, abs=0.04)


@pytest.mark.parametrize(
    ("mutual_information", "message"),
    [
        ([[0.0, 0.1, 0.2]], "square"),
        ([[0.0]], "at least two"),
        ([[0.0, -0.1], [-0.1, 0.0]], "nonnegative"),
        ([[0.0, 0.1], [0.2, 0.0]], "symmetric"),
        ([[0.1, 0.1], [0.1, 0.0]], "diagonal"),
        ([[0.0, float("inf")], [float("inf"), 0.0]], "finite"),
    ],
)
def test_pairwise_correlated_gaussian_validates_mi_matrix(
    mutual_information: list[list[float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PairwiseCorrelatedGaussian(mutual_information)


def test_pairwise_correlated_gaussian_rejects_non_psd_correlation() -> None:
    correlations = torch.tensor(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.0],
            [0.9, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    mutual_information = -0.5 * torch.log1p(-(correlations**2))
    mutual_information.fill_diagonal_(0)

    with pytest.raises(ValueError, match="positive semidefinite"):
        PairwiseCorrelatedGaussian(mutual_information)


def test_pairwise_correlated_gaussian_validates_dimensions_and_indices() -> None:
    mutual_information = [[0.0, 0.1], [0.1, 0.0]]
    with pytest.raises(ValueError, match="dim"):
        PairwiseCorrelatedGaussian(mutual_information, dim=0)
    with pytest.raises(ValueError, match="num_samples"):
        PairwiseCorrelatedGaussian(mutual_information, num_samples=-1)

    dataset = PairwiseCorrelatedGaussian(mutual_information, num_samples=2)
    with pytest.raises(IndexError):
        dataset[2]


def test_latent_pairwise_correlated_gaussian_batch_and_contexts() -> None:
    torch.manual_seed(11)
    mutual_information = torch.full((3, 3), 0.2)
    mutual_information.fill_diagonal_(0)
    dataset = LatentPairwiseCorrelatedGaussian(
        count=8,
        batch_size=5,
        mutual_information=mutual_information,
        dim=2,
        num_batches=7,
    )

    batch = dataset[0]

    assert len(dataset) == 7
    assert batch.batch_size == torch.Size([5])
    assert batch["x"].shape == (5, 3, 2)
    assert batch["z"].shape == (5, 2)
    assert batch["component_index"].shape == (5,)
    assert batch["component_index"].unique().numel() == 5
    assert torch.allclose(
        dataset.factor @ dataset.factor.T,
        dataset.correlation,
        atol=1e-6,
    )
    reconstructed = dataset.residual_correlation + dataset.context_strength
    assert torch.allclose(reconstructed, dataset.correlation, atol=1e-6)
    assert torch.allclose(
        dataset.residual_factor @ dataset.residual_factor.T,
        dataset.residual_correlation,
        atol=1e-6,
    )


def test_latent_pairwise_correlated_gaussian_preserves_marginal_mi() -> None:
    torch.manual_seed(13)
    target = 0.25
    dim = 2
    mutual_information = torch.full((3, 3), target)
    mutual_information.fill_diagonal_(0)
    dataset = LatentPairwiseCorrelatedGaussian(
        count=10,
        batch_size=5,
        mutual_information=mutual_information,
        dim=dim,
        num_batches=1_600,
    )
    samples = []
    for index in range(len(dataset)):
        batch = dataset[index]
        samples.append(batch["x"])
    x = torch.cat(samples)

    for i in range(3):
        for j in range(i + 1, 3):
            correlation = torch.corrcoef(
                torch.stack((x[:, i].flatten(), x[:, j].flatten()))
            )[0, 1]
            empirical_mi = -dim / 2 * torch.log1p(-(correlation**2))
            assert empirical_mi.item() == pytest.approx(target, abs=0.04)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0, "batch_size": 2}, "count"),
        ({"count": 3, "batch_size": 1}, "at least two"),
        ({"count": 2, "batch_size": 3}, "exceed count"),
        ({"count": 3, "batch_size": 2, "dim": 0}, "dim"),
        ({"count": 3, "batch_size": 2, "num_batches": -1}, "num_batches"),
        (
            {"count": 3, "batch_size": 2, "context_fraction": 1.1},
            "context_fraction",
        ),
    ],
)
def test_latent_pairwise_correlated_gaussian_validates_constructor(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LatentPairwiseCorrelatedGaussian(
            mutual_information=[[0.0, 0.1], [0.1, 0.0]],
            **kwargs,
        )


def test_latent_pairwise_correlated_gaussian_validates_indices() -> None:
    dataset = LatentPairwiseCorrelatedGaussian(
        count=3,
        batch_size=2,
        mutual_information=[[0.0, 0.1], [0.1, 0.0]],
        num_batches=2,
    )

    assert dataset[-1]["x"].shape == (2, 2, 1)
    with pytest.raises(IndexError):
        dataset[2]
